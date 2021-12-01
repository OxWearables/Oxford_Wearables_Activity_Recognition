# Oxford CDT/HDS Data Challenge: Activity recognition on the Capture-24 dataset

## Setup instructions in the VMs
1. Load and initialize Anaconda. This needs to be done only once (you may not need to run this if you already see `(bash)` written in front of your prompt).

   ```bash
   module load Anaconda3
   conda init bash
   ```
   Exit and re-login so that the above takes effect.
3. Create an anaconda environment from the provided requirements YAML file:
   ```bash
   conda env create -f environment.yml
   ```
   If you face issues with this, maybe try editing the last line of
   `environemnt.yml` where it says `prefix: ~/anaconda3/envs/cdt_wearables` and make it point to your
   Anaconda installation.
4. You are now ready to use the environment:
   ```bash
   conda activate cdt_wearables
   ```
   In future logins, you only need to run this last command.

## Datasets

The data required to run the notebooks can be found in
`/cdtshared/wearables/`.
Don't copy the data to your own `$HOME` to avoid blowing up the storage.
Instead, change the absolute paths in the notebooks where necessary.
Or better, create a soft link:
```bash
ln -s /cdtshared/wearables/capture24/ capture24  # create shortcut in current location
ln -s /cdtshared/wearables/processed_data/ processed_data
```

The folder `capture24/` contains CSV files for the accelerometer recordings,
with header `time,x,y,z,annotation`. The folder `processed_data/` contains numpy
arrays `X.npy`, `Y.npy`, `T.npy` and `pid.npy` that represent the same data but
chunked into windows of 30 sec. See the intro notebook for more details.

## Evaluation
On the presentation day, show `metrics.classification_report`,
`metrics.cohen_kappa`, and `metrics.f1_score(..., average='macro')` for your
model under 5-fold cross-validation (CV). Remember to do this using the
participant IDs. Also, if you used 5-fold CV for hyperparameter
tuning, you must re-randomize the 5-fold splits for the final scores. Report the
mean, min and max of the CV scores, e.g. `.75 (.61, .82)`.

## How to run Jupyter notebooks remotely

If you're using VSCode, it has a nice extension for [remote development](https://code.visualstudio.com/docs/remote/ssh) as well as support for [Jupyter notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks). Check if your editor has these features. Otherwise, see the following instructions:

1. In your remote machine, launch a Jupyter notebook with a specified port, e.g. 9000:
   ```bash
   jupyter-notebook --no-browser --port=9000
   ```
   This will output something like:
   ```bash
   To access the notebook, open this URL:
   http://localhost:9000/?token=
   b3ee74d492a6348430f3b74b52309060dcb754e7bf3d6ce4
   ```

1. On your local machine, perform port-forwarding, e.g. the following forwards the remote port 9000 to the local port 8888:
   ```bash
   ssh -N -f -L localhost:8888:localhost:9000 username@remote_address
   ```
   Note: You can use the same port numbers for both local and remote.

1. Finally, copy the URL from step 1. Then in your local machine, open
Chrome and paste the URL, but change the port to the local port (or do nothing else if you used the same port).
You should be able see the notebooks now.

Source code for the notebooks can be found in `src/` in case you prefer to work with pure Python instead.
