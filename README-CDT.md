# Oxford Activity Recognition Workshop: Instructions for Oxford CDT students

## Getting started

1. Load and initialize Anaconda. This needs to be done only once (you may not need to run this if you already see `(base)` written in front of your prompt).

   ```console
   module load Anaconda3
   conda init bash
   ```
   Exit and re-login so that the above takes effect.
1. Create a conda environment called `wearables-workshop` with minimal requirements (Python, git, Java):
    ```console
    conda create -n wearables-workshop python=3.9 git pip openjdk
    ```
1. Activate the environment:
    ```console
    conda activate wearables-workshop
    ```
    Your prompt should now show `(wearables-workshop)`.
1. Download workshop materials:
    ```console
    git clone https://github.com/OxWearables/Oxford_Wearables_Activity_Recognition.git
    ```
1. Navigate to workshop directory:
    ```console
    cd Oxford_Wearables_Activity_Recognition/
    ```
1. Install requirements for this workshop:
    ```console
    pip install -r requirements.txt
    ```
1. Launch notebook:
    ```console
    jupyter notebook
    ```

## Datasets

The data required for this workshop can be found in the shared path
`/cdtshared/wearables/`, so **you don't need to download it again in the notebooks**.
Do NOT copy the data to your `$HOME` directory to avoid blowing up the storage.
Instead, change the absolute paths in the corresponding notebooks, or create a soft link:
```console
cd  # go to $HOME
ln -s /cdtshared/wearables/capture24/ capture24  # create shortcut named 'capture24' in current directory
ln -s /cdtshared/wearables/processed_data/ processed_data  # create shortcut named 'processed_data' in current directory
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
   ```console
   jupyter-notebook --no-browser --port=1234
   ```
   This will output something like:
   ```console
   To access the notebook, open this URL:
   http://localhost:1234/?token=
   b3ee74d492a6348430f3b74b52309060dcb754e7bf3d6ce4
   ```

1. In your **local machine**, perform port-forwarding, e.g. the following forwards the remote port 1234 to the local port 1234:
   ```console
   ssh -N -f -L 1234:localhost:1234 username@remote_address
   ```

Now you should be able to access the URL link obtained in step 1 and see the notebooks.

## Troubleshooting

#### I'm getting "insufficient memory" errors

This is likely due to having several notebooks open and running. Make sure to
shutdown the notebooks that are no longer in use: Go to main menu, select notebook and click "Shutdown". Note that just closing the tab is not enough.

