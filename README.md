# Oxford Activity Recognition Workshop

## Getting started

```bash
# download this repo
$ git clone https://github.com/OxWearables/Oxford_Wearables_Activity_Recognition.git
# change into repo directory
$ cd Oxford_Wearables_Activity_Recognition/
# install environment with necessary dependencies
$ conda env create -f environment.yml
# or on a Mac install environment with necessary dependencies
$ conda env create -f environment_macos.yml
# activate environment
$ conda activate wearables_workshop
# launch notebook
$ jupyter notebook
```

The last command above will let you open the notebooks for this workshop in a
web browser. You can then start by clicking on `0_Intro.ipynb`.

## Troubleshooting

#### I'm getting "insufficient memory" errors

This is likely due to having several notebooks open and running. Make sure to
shutdown the notebooks that are no longer in use: Go to main menu, select notebook and click "Shutdown". Note that just closing the tab is not enough.

## License
See [license](LICENSE.md) before using these materials.
