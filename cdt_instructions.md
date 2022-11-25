# Oxford Activity Recognition Workshop: Instructions for CDT students

## Setup instructions in the VMs
1. Load and initialize Anaconda. This needs to be done only once (you may not need to run this if you already see `(base)` written in front of your prompt).

   ```bash
   module load Anaconda3
   conda init bash
   ```
   Exit and re-login so that the above takes effect.
1. Download this repo:
   ```bash
   git clone https://github.com/activityMonitoring/Oxford_Wearables_Activity_Recognition.git
   cd Oxford_Wearables_Activity_Recognition/
   ```
1. Create an anaconda environment using the provided YAML file:
   ```bash
   conda env create -f environment.yml
   ```
   Anaconda installation.
1. Finally, activate the environment:
   ```bash
   conda activate wearables_workshop
   ```
   In future logins, you only need to run this last command.

## Datasets

The data required to run the notebooks can be found in the shared path
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

If you're using VSCode, it has a nice extension for [remote
development](https://code.visualstudio.com/docs/remote/ssh) as well as support
for [Jupyter
notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
Check if your editor has these features. Otherwise, see the following
instructions:

1. In your **remote machine**, launch a Jupyter notebook with a specified port, e.g. 1234:
   ```bash
   jupyter-notebook --no-browser --port=1234
   ```
   This will output something like:
   ```bash
   To access the notebook, open this URL:
   http://localhost:1234/?token=
   b3ee74d492a6348430f3b74b52309060dcb754e7bf3d6ce4
   ```

1. In your **local machine**, perform port-forwarding, e.g. the following forwards the remote port 1234 to the local port 1234:
   ```bash
   ssh -N -f -L 1234:localhost:1234 username@remote_address
   ```
   Note: You can use same or different port numbers for local and remote.

Now you should be able to access the URL link obtained in step 1 and see the notebooks.

BTW source code for the notebooks can be found in `src/` in case you prefer to work with pure Python instead.
