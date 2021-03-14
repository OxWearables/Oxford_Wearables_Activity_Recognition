# %% [markdown]
'''
# Activity recognition on the Capture24 dataset

<img src="wrist_accelerometer.jpg" width="300"/>

The Capture24 dataset consists of wrist-worn accelerometer data
collected from about 150 participants. 
Along with the accelerometer, each participant wore a body camera during
daytime, and used a sleep diary to register their sleep and wake-up times.
The total wear time per participant was about 24 hours.
The accelerometer mearures acceleration in the three axes (x, y, z), and has
a sampling rate of 100Hz.
The body camera has a temporal resolution of 1 picture every 20 seconds.
Human annotators annotated the accelerometer timestamps by inspecting
the body camera images to infer the activities performed, and the sleep
diary for the sleep periods.

## Setup
'''

# %%
import os
import time
import re
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import manifold
from sklearn import metrics
from tqdm.auto import tqdm

# For reproducibility
np.random.seed(42)

# %% [markdown]
'''
## Load and inspect the dataset
'''

# %%

# Path to capture24 dataset
CAPTURE24_PATH = 'capture24/'

# Let's see what's in it
print(f'Content of {CAPTURE24_PATH}')
print(os.listdir(CAPTURE24_PATH))

# Let's load and inspect one participant
data = pd.read_pickle(CAPTURE24_PATH+'040.pkl')
print(f'\nParticipant 040:')
print(data)

print('\nWith NaNs removed...')
data.dropna(inplace=True)
print(data)

print("\nUnique annotations")
print(pd.Series(data['annotation'].unique()))

# %% [markdown]
'''
The annotations are based on the [Compendium of Physical
Activity](https://sites.google.com/site/compendiumofphysicalactivities/home).
There are more than 100 unique annotations identified in the whole dataset.
As you can see, the annotations can be very detailed. 

For our purposes, it is enough to translate the annotations into a simpler
set of labels. The provided *annotation-label-dictionary.csv* file contains
a few options that were used in previous works.
'''

# %%
ANNO_LABEL_DICT_PATH = 'annotation-label-dictionary.csv'

anno_label_dict = pd.read_csv(ANNO_LABEL_DICT_PATH, dtype='string')
print("Annotation-Label Dictionary")
print(anno_label_dict)

# Translate annotations using Willetts' labels
anno_label_dict.set_index('annotation', inplace=True)
data['label'] = anno_label_dict.loc[data['annotation'], 'label:Willetts2018'].values

print('\nLabel distribution (Willetts)')
print(data['label'].value_counts())

# %% [markdown]
'''
To continue, let's extract 30-sec windows of activity &mdash; these will make up
the learning dataset.
'''

# %%

def splitby_timegap(data, gap=1):
    split_id = (data.index.to_series().diff() > pd.Timedelta(gap, 'S')).cumsum()
    splits = data.groupby(by=split_id)
    return splits

def pick_majority(alist, criteria_fn):
    crit = [criteria_fn(_) for _ in alist]
    unq, cnt = np.unique(crit, return_counts=True)
    if len(cnt) < 1:
        return []
    maj = unq[np.argmax(cnt)]
    return [a for a, b in zip(alist, crit) if b == maj]

def extract_windows(data, window_len, stack=True):
    X, Y, T = [], [], []
    for _, data_split in splitby_timegap(data):
        for _, window in data_split.groupby(pd.Grouper(freq=window_len)):
            X.append(window[['x','y','z']].values)
            Y.append(window['label'].values)
            T.append(window.index.values)
    # Discard windows of irregular length (pick majority length)
    X, Y, T = pick_majority(X, len), pick_majority(Y, len), pick_majority(T, len)
    # Pick majority label in each window
    Y = [stats.mode(_)[0].item() for _ in Y]
    # Only keep window start time
    T = [_[0] for _ in T if len(_)>1]

    if stack:
        try:
            X, Y, T = np.stack(X), np.stack(Y), np.stack(T)
        except ValueError:
            X, Y, T = np.asarray(X), np.asarray(Y), np.asarray(T)
    return X, Y, T
    
# Extract 30s windows
X, Y, T = extract_windows(data, window_len='30s')
print('X shape:', X.shape)
print('Y shape:', Y.shape)
print('T shape:', T.shape)
print('\nLabel distribution (windowed)')
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
From the plots, it seems it should be pretty easy to classify "sleep"
and maybe "sit-stand".
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
The "sleep" dots are well clustered together, which supports our
guess that it should be easier to classify.

## t-SNE visualization
What if we want to visualize more components? A popular high-dimensional data
visualization tool is _t-distributed stochastic neighbor embedding_ (t-SNE).
Let's visualize 64 principal components.

*Note: this may take a while*
'''

# %%
print("Plotting t-SNE on 64 PCA components...")
pca = decomposition.PCA(n_components=64)  # 64 components this time
X_pca = pca.fit_transform(X_scaled)
tsne = manifold.TSNE(n_components=2,  # project down to 2 components
    init='random', random_state=42, perplexity=100)
X_tsne_pca = tsne.fit_transform(X_pca)
scatter_plot(X_tsne_pca, Y)

# %% [markdown]
'''
# Feature extraction
Let's extract some commonly used timeseries features from each activity
window. Feel free to engineer your own features!
'''

# %%

def extract_features(xyz):
    ''' Extract timeseries features. xyz is a window of shape (N,3) '''
    
    feats = {}
    feats['xMean'], feats['yMean'], feats['zMean'] = np.mean(xyz, axis=0)
    feats['xStd'], feats['yStd'], feats['zStd'] = np.std(xyz, axis=0)
    feats['xRange'], feats['yRange'], feats['zRange'] = np.ptp(xyz, axis=0)
    feats['xIQR'], feats['yIQR'], feats['zIQR'] = stats.iqr(xyz, axis=0)

    x, y, z = xyz.T

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
        feats['xyCorr'] = np.nan_to_num(np.corrcoef(x, y)[0,1])
        feats['yzCorr'] = np.nan_to_num(np.corrcoef(y, z)[0,1])
        feats['zxCorr'] = np.nan_to_num(np.corrcoef(z, x)[0,1])

    m = np.linalg.norm(xyz, axis=1)

    feats['mean'] = np.mean(m)
    feats['std'] = np.std(m)
    feats['range'] = np.ptp(m)
    feats['iqr'] = stats.iqr(m)
    feats['mad'] = stats.median_abs_deviation(m)
    feats['kurt'] = stats.kurtosis(m)
    feats['skew'] = stats.skew(m)

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
    init='random', random_state=42, perplexity=100)
X_tsne_feats = tsne.fit_transform(X_feats)
scatter_plot(X_tsne_feats, Y)

# %% [markdown]
'''
# Activity classification
Let's train a balanced random forest on the extracted features to perform
activity classification. We use the implementation from
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

print('\nClassifier performance in training set')
print(metrics.classification_report(Y, clf.predict(X_feats)))

# %% [markdown]
'''
The classification in-sample is just acceptable. This suggests
that we might need to add more discriminative features. Let's load another
subject to test and get the true (out-of-sample) performance.
'''

# %%

# Load another participant data
data2 = pd.read_pickle(CAPTURE24_PATH+'077.pkl').dropna()
# Translate annotations
data2['label'] = anno_label_dict.loc[data2['annotation'], 'label:Willetts2018'].values
# Extract 30s windows
X2, Y2, T2 = extract_windows(data2, window_len='30s')
# Extract features
X2_feats = pd.DataFrame([extract_features(x) for x in X2])

print('\nClassifier performance on held-out subject')
print(metrics.classification_report(Y2, clf.predict(X2_feats)))

# %% [markdown]
'''
The overall classification performance is much worse on the held-out subject.
As we expected, "sleep" classification remains quite good (f1-score of 0.90).

### Next steps
So far we've only looked at one subject. To use the whole dataset, repeat the
procedure per subject and concatenate the data, but beware of memory
issues. Below is a sample code that uses `np.memmap` to store the windows
directly onto disk. Re-run this notebook on more subjects, explore different
labels and define your own by modifying the file
*annotation-label-dictionary.csv*.

To save you some time, we have already extracted the dataset with
Willetts2018 labels and saved it in `dataset/`. But if you plan to use your
own annotation-label scheme or change the window lengths, then you'll need to
adapt and run the code below.

'''

# %%

def multi_extract_windows(outdir):

    X_DTYPE = 'f8'
    Y_DTYPE = 'U20'
    T_DTYPE = 'datetime64[ns]'
    P_DTYPE = 'U3'
    X_ROWSHAPE = (3000,3)

    n_old = n_new = 0
    is_first = True

    X_path = os.path.join(outdir, 'X.dat')  # windows
    Y_path = os.path.join(outdir, 'Y.dat')  # window labels
    T_path = os.path.join(outdir, 'T.dat')  # window start times
    P_path = os.path.join(outdir, 'P.dat')  # participant identifiers

    for datafile in tqdm(os.listdir(CAPTURE24_PATH)):

        mode = 'r+'
        if is_first:
            is_first = False
            mode = 'w+'

        data = pd.read_pickle(CAPTURE24_PATH+datafile).dropna()
        data['label'] = anno_label_dict.loc[data['annotation'], 'label:Willetts2018'].values
        X, Y, T = extract_windows(data, window_len='30s')

        # Skip to next loop if empty
        if len(X)<1 or len(Y)<1 or len(T)<1 :
            continue

        # Get participant ID from filename
        part = re.match(r'(\d{3})', datafile).group(1)
        P = np.repeat([part], len(X))

        n_new = n_old + len(X)
        Xmap = np.memmap(X_path, mode=mode, dtype=X_DTYPE, shape=(n_new,) + X_ROWSHAPE)
        Ymap = np.memmap(Y_path, mode=mode, dtype=Y_DTYPE, shape=(n_new,))
        Tmap = np.memmap(T_path, mode=mode, dtype=T_DTYPE, shape=(n_new,))
        Pmap = np.memmap(P_path, mode=mode, dtype=P_DTYPE, shape=(n_new,))
        Xmap[n_old:], Ymap[n_old:], Tmap[n_old:], Pmap[n_old:] = X, Y, T, P
        Xmap.flush(); Ymap.flush(), Tmap.flush(), Pmap.flush()
        n_old = n_new

    X_shape = (n_new,) + X_ROWSHAPE
    Y_shape = T_shape = P_shape = (n_new,)

    info = { 'X_shape': X_shape, 'X_dtype': X_DTYPE,
             'Y_shape': Y_shape, 'Y_dtype': Y_DTYPE,
             'T_shape': T_shape, 'T_dtype': T_DTYPE,
             'P_shape': P_shape, 'P_dtype': P_DTYPE }

    with open(os.path.join(outdir, 'info.json'), 'w') as f:
        json.dump(info, f)

    print(f'Output files saved in {outdir}')
    print(os.listdir(outdir))

    print('X shape:', Xmap.shape)
    print('Y shape:', Ymap.shape)
    print('T shape:', Tmap.shape)
    print('P shape:', Pmap.shape)

    return info
    
# # Extract windows and save in dataset/
# OUTDIR = 'dataset/'
# os.system(f'mkdir -p {OUTDIR}')
# multi_extract_windows(OUTDIR)

# # Check that extraction worked
# print('\nReloading memmap data')
# with open(OUTDIR+'info.json', 'r') as f:
#     info = json.load(f)  # load metadata
# X = np.memmap(OUTDIR+'X.dat', mode='r', dtype=info['X_dtype'], shape=tuple(info['X_shape']))
# Y = np.memmap(OUTDIR+'Y.dat', mode='r', dtype=info['Y_dtype'], shape=tuple(info['Y_shape']))
# T = np.memmap(OUTDIR+'T.dat', mode='r', dtype=info['T_dtype'], shape=tuple(info['T_shape']))
# P = np.memmap(OUTDIR+'P.dat', mode='r', dtype=info['P_dtype'], shape=tuple(info['P_shape']))
# print('X shape:', X.shape)
# print('Y shape:', Y.shape)
# print('T shape:', T.shape)
# print('P shape:', P.shape)

# %% [markdown]
'''
## References
- Ideas for hand-crafted features:
    - [Physical activity classification using the GENEA wrist-worn accelerometer](https://www.ncbi.nlm.nih.gov/pubmed/21988935)
    - [A universal, accurate intensity-based classification of different physical activities using raw data of accelerometer](https://www.ncbi.nlm.nih.gov/pubmed/24393233)
    - [Activity recognition using a single accelerometer placed at the wrist or ankle](https://www.ncbi.nlm.nih.gov/pubmed/23604069)
    - [Hip and Wrist Accelerometer Algorithms for Free-Living Behavior Classification](https://www.ncbi.nlm.nih.gov/pubmed/26673126)

- Papers using the capture24 dataset:
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
