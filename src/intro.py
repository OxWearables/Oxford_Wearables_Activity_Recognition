# %% [markdown]
'''
# Activity recognition on the Capture-24 dataset

<img src="wrist_accelerometer.jpg" width="300"/>

The Capture-24 dataset contains wrist-worn accelerometer data
collected from 151 participants. To obtain ground truth annotations, the
participants also wore a body camera during daytime, and used sleep diaries to
register their sleep times. Each participant was recorded for roughly 24 hours.
The accelerometer was an Axivity AX3 wrist watch (image above) that mearures
acceleration in all three axes (x, y, z) at a sampling rate of 100Hz.
The body camera was a Vicon Autographer with a sampling rate of 1 picture every 20 seconds.
Note that the camera images are not part of the data release &mdash; only the
raw acceleration trace with text annotations are provided.

## Setup
'''

# %%
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import manifold
from sklearn import metrics
from tqdm.auto import tqdm
from joblib import Parallel, delayed

import utils

# For reproducibility
np.random.seed(42)

# %% [markdown]
'''
## Load and inspect the dataset

To run this notebook, you'll need the
[Capture-24 dataset](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001).

'''

# %%

# Path to capture24 dataset
CAPTURE24_PATH = 'capture24/'

# Let's see what's in it
print(f'Content of {CAPTURE24_PATH}')
print(os.listdir(CAPTURE24_PATH))

# Let's load and inspect one participant
data = utils.load_data(CAPTURE24_PATH+'P001.csv.gz')
print('\nParticipant P001:')
print(data)

print("\nAnnotations in P001")
print(pd.Series(data['annotation'].unique()))

# %% [markdown]
'''
The annotations are based on the [Compendium of Physical
Activity](https://sites.google.com/site/compendiumofphysicalactivities/home).
In total, there were more than 200 distinct annotations in the whole dataset.
As you can see, the annotations can be very detailed.

To develop a model for activity recognition, let's chunk the data into windows of
30 sec. The activity recognition model will then be trained to classify the
individual windows.
'''

#%%

X, Y, T = utils.make_windows(data, winsec=30)
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("T shape:", T.shape)

#%% [markdown]
'''
As mentioned, there can be hundreds of distinct annotations, many of which are
very similar (e.g. "sitting, child care", "sitting, pet care").
For our purposes, it is enough to translate the annotations into a simpler
set of labels. The provided file *annotation-label-dictionary.csv*
contains different annotation-to-label mappings that can be used.
'''

# %%
ANNO_LABEL_DICT_PATH = CAPTURE24_PATH+'annotation-label-dictionary.csv'
anno_label_dict = pd.read_csv(ANNO_LABEL_DICT_PATH, index_col='annotation', dtype='string')
print("Annotation-Label Dictionary")
print(anno_label_dict)

# Translate annotations using Willetts' labels  (see paper reference at the bottom)
Y = anno_label_dict.loc[Y, 'label:Willetts2018'].to_numpy()

print('\nLabel distribution (Willetts)')
print(pd.Series(Y).value_counts())

# %% [markdown]
'''
We observe some imbalance in the data. This will likely be an issue
later for the machine learning model.

# Visualization
Visualization helps us get some insight and anticipate the difficulties that
may arise during the modelling.
Let's visualize some examples for each activity label.
'''

# %%
NPLOTS = 5
unqY = np.unique(Y)
fig, axs = plt.subplots(len(unqY), NPLOTS, sharex=True, sharey=True, figsize=(10,10))
for y, row in zip(unqY, axs):
    idxs = np.random.choice(np.where(Y==y)[0], size=NPLOTS)
    row[0].set_ylabel(y)
    for x, ax in zip(X[idxs], row):
        ax.plot(x)
        ax.set_ylim(-5,5)
fig.tight_layout()
fig.show()

# %% [markdown]
'''
From the plots, we can already tell it should be easier to classify "sleep"
and maybe "sit-stand", with the signal variance being a good discriminative
feature for this.
Next, let's try to visualize the data in a scatter-plot.
The most standard approach to visualize high-dimensional points is to
scatter-plot the first two principal components of the data.

## PCA visualization

'''

# %%

def scatter_plot(X, Y):
    unqY = np.unique(Y)
    fig, ax = plt.subplots()
    for y in unqY:
        X_y = X[Y==y]
        ax.scatter(X_y[:,0], X_y[:,1], label=y, alpha=.5, s=10)
    fig.legend()
    fig.show()

print("Plotting first two PCA components...")
scaler = preprocessing.StandardScaler()  # PCA requires normalized data
X_scaled = scaler.fit_transform(X.reshape(X.shape[0],-1))
pca = decomposition.PCA(n_components=2)  # two components
X_pca = pca.fit_transform(X_scaled)
scatter_plot(X_pca, Y)

# %% [markdown]
'''
## t-SNE visualization
PCA's main limitation is in dealing with data that is not linearly separable.
Another popular high-dimensional data visualization tool is _t-distributed
stochastic neighbor embedding_ (t-SNE).  Let's first use it on top of PCA to
visualize 50 principal components.

*Note: this may take a while*
'''

# %%
print("Plotting t-SNE on 50 PCA components...")
pca = decomposition.PCA(n_components=50)  # 64 components this time
X_pca = pca.fit_transform(X_scaled)
tsne = manifold.TSNE(n_components=2,  # project down to 2 components
    init='random', random_state=42, perplexity=100, learning_rate='auto')
X_tsne_pca = tsne.fit_transform(X_pca)
scatter_plot(X_tsne_pca, Y)

# %% [markdown]
'''
# Feature extraction
Let's extract a few signal features for each window.
Feel free to engineer your own features!
'''

# %%

def extract_features(xyz):
    ''' Extract features. xyz is an array of shape (N,3) '''

    feats = {}
    feats['xMean'], feats['yMean'], feats['zMean'] = np.mean(xyz, axis=0)
    feats['xStd'], feats['yStd'], feats['zStd'] = np.std(xyz, axis=0)
    v = np.linalg.norm(xyz, axis=1)  # magnitude stream
    feats['mean'], feats['std'] = np.mean(v), np.std(v)

    return feats

X_feats = pd.DataFrame([extract_features(x) for x in X])
print(X_feats)

# %% [markdown]
'''
Let's visualize the data again using t-SNE, but this time using the extracted
features rather than the principal components.

*Note: this may take a while*
'''

# %%
print("Plotting t-SNE on extracted features...")
tsne = manifold.TSNE(n_components=2,
    init='random', random_state=42, perplexity=100, learning_rate='auto')
X_tsne_feats = tsne.fit_transform(X_feats)
scatter_plot(X_tsne_feats, Y)

# %% [markdown]
'''
# Activity classification
Le fun part. Let's train a balanced random forest on the extracted features to
perform activity classification. We use the implementation from
[`imbalanced-learn`](https://imbalanced-learn.org/stable/) package, which has
better support for imbalanced datasets.
'''

# %%
clf = BalancedRandomForestClassifier(
    n_estimators=1000,
    replacement=True,
    sampling_strategy='not minority',
    n_jobs=4,
    random_state=42,
)
clf.fit(X_feats, Y)
Y_pred = clf.predict(X_feats)

#%%
print('\nClassifier performance in training set')
print(metrics.classification_report(Y, Y_pred, zero_division=0))

fig, axs = utils.plot_compare(T, Y, Y_pred, trace=X_feats['std'])
fig.show()

# %% [markdown]
'''
The classification performance is very good, but this is in-sample! Let's load
another subject to test and get the true (out-of-sample) performance.
'''

# %%

# Load another participant
data2 = utils.load_data(CAPTURE24_PATH+'P002.csv.gz')
X2, Y2, T2 = utils.make_windows(data2, winsec=30)
Y2 = anno_label_dict.loc[Y2, 'label:Willetts2018'].to_numpy()
X2_feats = pd.DataFrame([extract_features(x) for x in X2])
Y2_pred = clf.predict(X2_feats)

print('\nClassifier performance on held-out subject')
print(metrics.classification_report(Y2, Y2_pred, zero_division=0))

fig, axs = utils.plot_compare(T2, Y2, Y2_pred, trace=X2_feats['std'])
fig.show()

# %% [markdown]
'''
As expected, the classification performance is much worse out of sample, with
the macro-averaged F1-score dropping from .90 to .37.
On the other hand, the scores for the easy classes "sleep" and "sit-stand" remained good.
Finally, note that participant P001 didn't have the "bicycling" class while
participant P002 didn't have the "vehicle" class.

### Next steps
So far we've only trained on one subject. To use the whole dataset, repeat the
data processing for each of the subjects and concatenate them. You can use the
code below for this.

'''

#%%

def load_all_and_make_windows(datafiles):

    def worker(datafile):
        X, Y, T = utils.make_windows(utils.load_data(datafile), winsec=30)
        pid = os.path.basename(datafile).split(".")[0]  # participant ID
        pid = np.asarray([pid] * len(X))
        return X, Y, T, pid

    results = Parallel(n_jobs=4)(
        delayed(worker)(datafile) for datafile in tqdm(datafiles))

    X = np.concatenate([result[0] for result in results])
    Y = np.concatenate([result[1] for result in results])
    T = np.concatenate([result[2] for result in results])
    pid = np.concatenate([result[3] for result in results])

    return X, Y, T, pid

# # Uncomment below to process all files
# DATAFILES = CAPTURE24_PATH+'P[0-9][0-9][0-9].csv.gz'
# X, Y, T, pid = load_all_and_make_windows(glob(DATAFILES))
# # Save arrays for future use
# os.makedirs("processed_data/", exist_ok=True)
# np.save("processed_data/X.npy", X)
# np.save("processed_data/Y.npy", Y)
# np.save("processed_data/T.npy", T)
# np.save("processed_data/pid.npy", pid)

# %% [markdown]
'''
## References
**Feature extraction**

- [On the role of features in human activity recognition](https://dl.acm.org/doi/10.1145/3341163.3347727)
- [A Comprehensive Study of Activity Recognition Using Accelerometers](https://www.mdpi.com/2227-9709/5/2/27)

**Papers using the Capture-24 dataset**

- [Reallocating time from machine-learned sleep, sedentary behaviour or
light physical activity to moderate-to-vigorous physical activity is
associated with lower cardiovascular disease
risk](https://www.medrxiv.org/content/10.1101/2020.11.10.20227769v2.full?versioned=true)
(Walmsley2020 labels)
- [GWAS identifies 14 loci for device-measured
physical activity and sleep
duration](https://www.nature.com/articles/s41467-018-07743-4)
(Doherty2018 labels)
- [Statistical machine learning of sleep and physical activity phenotypes
from sensor data in 96,220 UK Biobank
participants](https://www.nature.com/articles/s41598-018-26174-1)
(Willetts2018 labels)

'''
# %%
