# Activity recognition on Capture24 dataset

##### Requirements
The code for feature extraction is written in Java (distilled from
[here](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/blob/master/java/EpochWriter.java)). We need the following:

- Compile the Java feature extractor: `javac -cp JTransforms-3.1-with-dependencies.jar *.java`
- JPype, a package to use Java packages in Python: `conda install -c conda-forge jpype1`

##### Datasets

To run all the notebooks, you will need `capture24_small.npz` and `X_raw_small.npy`.
These can be found in `/well/doherty/projects/cdt_data_challenge` (Nov 05, 2019).
