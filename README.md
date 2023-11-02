# Oxford Activity Recognition Workshop

## Getting started

1. Download and install Anaconda or Miniconda: https://docs.conda.io/projects/miniconda/en/latest/
1. Create a conda environment called `wearables-workshop` with minimal requirements (Python, git, Java):
    ```console
    $ conda create -n wearables-workshop python=3.9 git pip openjdk
    ```
1. Activate the environment:
    ```console
    $ conda activate wearables-workshop
    ```
    Your prompt should now show `(wearables-workshop)`.
1. Download workshop materials:
    ```console
    $ git clone https://github.com/OxWearables/Oxford_Wearables_Activity_Recognition.git
    ```
1. Navigate to workshop directory:
    ```console
    $ cd Oxford_Wearables_Activity_Recognition/
    ```
1. Install requirements for this workshop:
    ```console
    $ pip install -r requirements.txt
    ```
1. Launch notebook:
    ```console
    $ jupyter notebook
    ```

The last command above will let you open the notebooks for this workshop in a
web browser. You can then start by clicking on `0_Intro.ipynb`.

After reboots, you can access the notebooks again with the following:
1. Activate the environment:
    ```console
    $ conda activate wearables-workshop
    ```
    Your prompt should now show `(wearables-workshop)`.
1. Navigate to workshop materials:
    ```console
    $ cd /path/to/Oxford_Wearables_Activity_Recognition/
    ```
1. Launch notebook:
    ```console
    $ jupyter notebook
    ```

## Troubleshooting

#### I'm getting "insufficient memory" errors

This is likely due to having several notebooks open and running. Make sure to
shutdown the notebooks that are no longer in use: Go to main menu, select notebook and click "Shutdown". Note that just closing the tab is not enough.

## License
See [license](LICENSE.md) before using these materials.
