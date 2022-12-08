# Deploying models on a cluster

This tutorial will walk you through the steps to run your model on the UK Biobank accelerometer data. 

This is done on the Biomedical Research Computing (BMRC) High Performance Computing (HPC) cluster. This is a cluster of hundreds of machines (nodes) and thousands of cores with a shared storage system. The high number of cores allows for many processes to run in parallel. 

The basic concept is very simple: your model will be run ('deployed') on each accelerometer file, but hundreds of time in parallel. This deployment will be handled by the cluster scheduler system (Slurm). All your python script needs to do is take 1 accelerometer file as the input, run your model, and save the result to disk.

## Getting started
You should have access to the BMRC cluster. Log in with the same credentials as the VM: 

`ssh <username>@cluster2.bmrc.ox.ac.uk`

Your workspace on BMRC is located in `/well/doherty/projects/cdt/users/<username>` (this folder is also available on the VM). Replace `<username>` with your username. Your code goes here. Start by cloning this repo:

```bash
cd /well/doherty/projects/cdt/users/<username>
git clone https://github.com/OxWearables/Oxford_Wearables_Activity_Recognition

# cd into the tutorial folder
cd Oxford_Wearables_Activity_Recognition/7_cluster_tutorial

# copy additional files to this folder
cp /well/doherty/projects/cdt/shared/write-BMRC-script.py ./
cp /well/doherty/projects/cdt/shared/ukb-short.txt ./
cp /well/doherty/projects/cdt/shared/sample.cwa.gz ./
```

## Hello World in parallel
We'll first illustrate parallel deployment on BMRC with a Hello World example.

Open the file `helloworld.py`. It contains the following script:

```python
import argparse
import socket
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int, help='a number to print')
    args = parser.parse_args()

    time.sleep(10)
    print(f'Hello World! I am task number {args.id} on host {socket.gethostname()}.')
```

`helloworld.py` takes a number as a command line argument, waits 10 seconds and then prints a Hello World line with the given number and the device hostname. The 10s sleep is there to simulate work being done. For example:

```bash
python helloworld.py 1

# (after 10s) Hello World! I am task number 1 on host rescomp1.hpc.in.bmrc.ox.ac.uk.
```

Now create a file `commands.txt` that contains the following lines:

```bash
python helloworld.py 1
python helloworld.py 2
python helloworld.py 3
python helloworld.py 4
python helloworld.py 5
```

This file contains all the commands that we want the cluster to execute. Each line is treated as a single task, and the cluster scheduler will distribute each task among its nodes. This is done in parallel, but it doesn't necessarily mean all 5 tasks will run at the same time - this depends on occupancy. In this small example with only 5 tasks, however, they are likely to be executed at the same time. 

Now use the script `write-BMRC-script.py` to generate a cluster submission script:

```bash
python write-BMRC-script.py commands.txt

# BMRC sbatch script written to: commands.sh
```

This generates the submission script `commands.sh`. You don't need to worry too much about how this script works. The main takeaway is that it takes the contents of `commands.txt` and submits them as parallel tasks to the cluster (a so called 'array job'). The output of each task will be saved to a logfile located in a newly created `logs` folder.

Submission is done with the `sbatch` command. **NOTE**: the cluster will use your *current directory* as the working directory when executing each task. It is therefore important that when you submit your job to the cluster, your shell is located in the folder where the `commands.sh`, `commands.txt` and your `helloworld.py` files are located (double check this with `ls`). Remember this for when you're running your own model: keep everything in the same directory and submit your job from there.

```bash
sbatch commands.sh
# Submitted batch job 9281943
```

Immediately after this, execute `sq`. You will see something like this:

```bash
     JOBID PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
 9281943_1     short commands   azw524  RUNNING       0:02 1-06:00:00      1 compa025
 9281943_2     short commands   azw524  RUNNING       0:02 1-06:00:00      1 compa038
 9281943_3     short commands   azw524  RUNNING       0:02 1-06:00:00      1 compe061
 9281943_4     short commands   azw524  RUNNING       0:02 1-06:00:00      1 compf005
 9281943_5     short commands   azw524  RUNNING       0:02 1-06:00:00      1 compe063
```

This shows your list of pending jobs. Note the same job id as the one returned from `sbatch`, with a task number appended. Each task corresponds to a line in `commands.txt`. Your tasks will be either `Pending` or `Running`. In the `Pending` state the scheduler is waiting for a free spot, while `Running` means your task is being executed. 

When your job is finished, you will find 5 log files in the `./logs` directory, 1 for each command in `commands.txt`:

```bash
ls ./logs
# commands-9281943_1.log
# commands-9281943_2.log
# commands-9281943_3.log
# commands-9281943_4.log
# commands-9281943_5.log
```

Inspect them:

```bash
more ./logs/* | cat

#::::::::::::::
#./logs/commands-9281943_1.log
#::::::::::::::
# 24/11/2022 14:03:28
# python helloworld.py 1
# Hello World! I am task number 1 on host compa025.hpc.in.bmrc.ox.ac.uk.
# CPU time : 0 min 10 sec
# 24/11/2022 14:03:38

#::::::::::::::
#./logs/commands-9281943_2.log
#::::::::::::::
# 24/11/2022 14:03:28
# python helloworld.py 2
# Hello World! I am task number 2 on host compa038.hpc.in.bmrc.ox.ac.uk.
# CPU time : 0 min 10 sec
# 24/11/2022 14:03:38

#::::::::::::::
#./logs/commands-9281943_3.log
#::::::::::::::
# 24/11/2022 14:03:28
# python helloworld.py 3
# Hello World! I am task number 3 on host compe061.hpc.in.bmrc.ox.ac.uk.
# CPU time : 0 min 10 sec
# 24/11/2022 14:03:38

#::::::::::::::
#./logs/commands-9281943_4.log
#::::::::::::::
# 24/11/2022 14:03:28
# python helloworld.py 4
# Hello World! I am task number 4 on host compf005.hpc.in.bmrc.ox.ac.uk.
# CPU time : 0 min 10 sec
# 24/11/2022 14:03:38

#::::::::::::::
#./logs/commands-9281943_5.log
#::::::::::::::
# 24/11/2022 14:03:28
# python helloworld.py 5
# Hello World! I am task number 5 on host compe063.hpc.in.bmrc.ox.ac.uk.
# CPU time : 0 min 10 sec
# 24/11/2022 14:03:38
```

Each file contains the command that's being executed, followed by the output and some timing information. Take note of the Hello World output with task numbers 1 through 5. You'll also notice that the hostname looks different from the machine you're currently on: this is the hostname of the node that executed your task. You may see different hostnames in each task, but it's also possible all 5 were executed on the same machine (but still in parallel, on different cores).

There's one more command that can be useful in the event you need to cancel a running job: `scancel`. This takes the job id that's returned from `sbatch`. So if you wanted to cancel the job from above before it finished, you could do `scancel 9281943`


## UK Biobank deployment

Deploying your model on the UK Biobank accelerometer data follows the exact same flow as the Hello World example above. A template script has been provided in `template.py` for this. The comments in this script will guide you through its usage. The key points are that `template.py` takes the path of an accelerometer file via the command line, reads and extracts the raw data, applies the model and saves the predictions to disk.

You should also have a working Anaconda environment for your model. Create it on BMRC with the usual methods.

### Model test
You should add your own model to `template.py` and test it first on the sample file that you copied in the previous step `sample.cwa.gz`. Do this on your VM (not on BMRC), or on your local machine. Your BMRC work folder is available on the VM under the same path `/well/doherty/projects/cdt/users/<username>`.

`python template.py sample.cwa.gz`

The output will be stored in the `output` folder. After confirming this works and produces the expected output, move on to the next step.

### UKB submission script

In `/well/doherty/projects/cdt/shared/` you can find files that contain the commands to deploy the template on the UKB accelerometer files:

- `ukb-short.txt` with the first 100 files for testing
- `ukb-full.txt` with all 103k files (this one will be made available in the first week of the data challenge)

You should have already copied `ukb-short.txt` to the tutorial folder in the previous part. Inspect this file with `head ukb-short.txt`. This file follows the same structure as the hello world example.

**Test your model first on `ukb-short`.** You still need to create the submission script yourself. This time also pass the name of your conda environment `wearables_workshop` (change this accordingly) and add a new `--batch` parameter:

```bash
python write-BMRC-script.py ukb-short.txt --batch 2 --conda wearables_workshop 

# BMRC sbatch script written to: ukb-short.sh
```

In the hello world example, each command was executed in 1 task. For UKB deployment, it's better to execute multiple commands per task. This improves efficiency of the job by spending less time waiting for a free slot. Submit your job with `sbatch ukb-short.sh`. Follow the progress with `sq`. You should see 50 tasks (because every task processes 2 files).

Reminder: `template.py` (with your model), `ukb-short.txt` and `ukb-short.sh` should all exist in the same folder, and that should be your current directory when submitting the job.

### Participant selection

After confirming your model works on `ukb-short.txt`, we can generate the final script to run it on the whole dataset. If you know beforehand that only a subset will have the medical outcome data that you're analysing, we can do one more filtering step to deploy the model only on this cohort. This is optional - if you're unsure of your cohort at the time of submission, you can use `ukb-full.txt`. But doing the participant selection will speed up the computation.

Using the RAP system, generate a text file `include.txt` that contains line-by-line the `eid` of the participants in your cohort (a csv file with 1 column, the `eid` header is optional):

```
eid
1234567
8901234
5678901
2345678
9012345
```

Make sure this file does **NOT** have an empty line at the end. 

Copy the file from the RAP to your user folder on the VM at `/well/doherty/projects/cdt/users/<username>` (see the instructions [here](https://oxwearables.github.io/wearable-teaching/prac5-6-association-analysis-in-uk-biobank#getting-files-between-the-vm-and-the-rap-using-the-command-line-interface)). Do **NOT** download this file to your local machine: the `eid` is patient information.

Then you can use `grep` to filter the entries in `ukb-full.txt` that match the include file:

```bash
grep -f include.txt ukb-full.txt > ukb-selection.txt
```

Inpect `ukb-selection.txt` to see if it only contains your cohort's participants. Now generate the final submission script from `ukb-selection.txt`. Also give a heads-up to your tutor before submitting. This time use a batch size of 25. This will take anywhere from a few hours to a day, depending on your model and cohort size, so it's best to do it overnight. We reserved capacity on BMRC for the full job, which is accessed with the `--reservation` parameter to `sbatch`. You can follow the progress with the `sq` command.

Tip: empty the `output` folder before submitting your job, so that it only contains the output files from your final deployment.

```bash
# delete output files from before (copy any files you need to keep to another folder!)
rm -r output/*

python write-BMRC-script.py ukb-selection.txt --batch 25 --conda wearables_workshop 

# use reserved capacity for the full job
sbatch --reservation=doherty_546 ukb-selection.sh

sq
```

## Summary file aggregation
After successful model deployment, the `output` folder will contain the predicted time series and summary files for each participant, located in their respective group folders (e.g.: `group1/123456.csv` and `group1/123456_summary.csv`). 

To simplify working with the summary files, we can aggregate the individual summary files together into 1 large `summary.csv` file, with one participant per row. Use the `merge_summary.py` script available in this repo. This script takes as argument the location of your `output` folder. You'll also need to install `dask` in your conda environment. 

Do this on the VM. Reminder: you don't need to copy the output files from BMRC to the VM, your BMRC work folder (that contains your model output) is available under `/well/doherty/projects/cdt/users/<username>` on both systems.

```bash
# do this on the VM
conda activate wearables_workshop
conda install dask

# cd into the folder that contains the output folder
# if you submitted your job from the tutorial folder, your output folder will be in:
cd /well/doherty/projects/cdt/users/<username>/Oxford_Wearables_Activity_Recognition/7_cluster_tutorial

# this can take a few minutes
python merge_summary.py ./output
# summary saved to ./output/summary.csv
```

You can then use the aggregated `summary.csv` file for your epi analysis.