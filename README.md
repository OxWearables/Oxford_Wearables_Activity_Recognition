# Oxford Activity Recognition Workshop

## Getting started

- Download and install Anaconda or Miniconda: https://docs.conda.io/projects/miniconda/en/latest/
- Create a virtual environment:
    ```bash
    # create a conda environment called 'wearables-workshop' with minimal requirements (Python, git, Java)
    $ conda create -n wearables-workshop python=3.9 git pip openjdk
    # activate the environment - prompt should now show (wearables-workshop)
    $ conda activate wearables-workshop
    # download workshop materials
    $ git clone https://github.com/OxWearables/Oxford_Wearables_Activity_Recognition.git
    # navigate to workshop directory
    $ cd Oxford_Wearables_Activity_Recognition/
    # install requirements for this workshop
    $ pip install -r requirements.txt
    # launch notebook
    $ jupyter notebook
    ```

The last command above will let you open the notebooks for this workshop in a
web browser. You can then start by clicking on `0_Intro.ipynb`.

After reboots, you can access the notebooks again with the following:
```bash
# activate the environment - prompt should now show (wearables-workshop)
$ conda activate wearables-workshop
# navigate to workshop materials
$ cd /path/to/Oxford_Wearables_Activity_Recognition/
# launch notebook
$ jupyter notebook
```

## Troubleshooting

#### I'm getting "insufficient memory" errors

This is likely due to having several notebooks open and running. Make sure to
shutdown the notebooks that are no longer in use: Go to main menu, select notebook and click "Shutdown". Note that just closing the tab is not enough.

## License
See [license](LICENSE.md) before using these materials.
