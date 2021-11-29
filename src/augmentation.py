# %% [markdown]
'''
# Data augmentation

Data augmentation is a straighforward way to artificially increase the size
of the dataset while embedding invariances into the model.

## Setup
'''
# %%
import os
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import metrics
from tqdm.auto import tqdm

# For reproducibility
np.random.seed(42)

# %% [markdown]
'''
## Load dataset
'''

# %%

# Path to your extracted windows
DATASET_PATH = 'processed_data/'
X_FEATS_PATH = 'X_feats.pkl'  # path to your extracted features, if have one
print(f'Content of {DATASET_PATH}')
print(os.listdir(DATASET_PATH))

X = np.load(DATASET_PATH+'X.npy', mmap_mode='r')
Y = np.load(DATASET_PATH+'Y.npy')
T = np.load(DATASET_PATH+'T.npy')
pid = np.load(DATASET_PATH+'pid.npy')
X_feats = pd.read_pickle('X_feats.pkl')

# As before, let's map the text annotations to simplified labels
ANNO_LABEL_DICT_PATH = 'capture24/annotation-label-dictionary.csv'
anno_label_dict = pd.read_csv(ANNO_LABEL_DICT_PATH, index_col='annotation', dtype='string')
Y = anno_label_dict.loc[Y, 'label:Willetts2018'].to_numpy()

# %% [markdown]
'''
## Train/test split
'''

# %%

# Hold out participants P101-P151 for testing (51 participants)
test_ids = [f'P{i}' for i in range(101,152)]
mask_test = np.isin(pid, test_ids)
mask_train = ~mask_test
X_train, Y_train, T_train, pid_train = \
    X_feats[mask_train], Y[mask_train], T[mask_train], pid[mask_train]
X_test, Y_test, T_test, pid_test = \
    X_feats[mask_test], Y[mask_test], T[mask_test], pid[mask_test]
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# %% [markdown]
'''
## Train a random forest classifier

*Note: this may take a while*
'''

# %%
clf = BalancedRandomForestClassifier(
    n_estimators=2000,
    replacement=True,
    sampling_strategy='not minority',
    n_jobs=4,
    random_state=42,
    verbose=1
)
clf.fit(X_train, Y_train)

Y_test_pred = clf.predict(X_test)
print('\nClassifier performance')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_pred, zero_division=0))

# %% [markdown]
'''
## Robustness to unforseen scenarios

What if the subjects in the test set wore the device differently from
those in the training set? For example, suppose that all the subjects in the
training set were right-handed, then the model could underperform on a test
subject who is left-handed. This would more or less result in the device having
been rotated. Another typical scenario happens when we want our model to be
deployable on other accelerometer devices with different axis orientations.

<img src="wrist_accelerometer.jpg" width="200"/>

Let's generate an artificial test set simulating this scenario by flipping two
of the axes signs (this may simulate a different device specs, but it does not
exactly simulate handedness since the movement dynamics are also mirrored, but
it is enough to demonstrate our point). For this, we will need to grab the raw
test data, rotate it, and re-compute the same features.

*Note: this may take a while*
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

# %%

print("Creating test set with 'rotated device'...")
X_raw_test = X[mask_test]
X_test_new = []
for i in tqdm(range(X_raw_test.shape[0])):
    # Rotate device
    x = X_raw_test[i].copy()
    x[:,1] *= -1
    x[:,2] *= -1
    X_test_new.append(extract_features(x))
X_test_new = pd.DataFrame(X_test_new)

# %% [markdown]
''' ### Performance on simulated test set '''

# %%

Y_test_new_pred = clf.predict(X_test_new)
print('\nClassifier performance -- simulated test set')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_new_pred, zero_division=0))

# %% [markdown]
'''
The model performance is notably worse on the simulated test set. The solution
is to simply augment the training dataset with the same rotation &mdash; we want
our model to perform well no matter how/what device was worn.

## Data augmentation

*Note: this may take a while*
'''

# %%
print("Creating training set with 'rotated device'...")
X_raw_train = X[mask_train]
X_train_new = []
for i in tqdm(range(X_raw_train.shape[0])):
    # Rotate device
    x = X_raw_train[i].copy()
    x[:,1] *= -1
    x[:,2] *= -1
    X_train_new.append(extract_features(x))
X_train_new = pd.DataFrame(X_train_new)

# Add the "new data" to training set
X_aug_train = pd.concat((X_train, X_train_new))
Y_aug_train = np.concatenate((Y_train, Y_train))
print("X_aug_train shape:", X_aug_train.shape)

# %% [markdown]
''' ### Re-train with augmented dataset

*Note: this may take a while*
'''

# %%
clf = BalancedRandomForestClassifier(
    n_estimators=2000,
    replacement=True,
    sampling_strategy='not minority',
    n_jobs=4,
    random_state=42,
    verbose=1
)
clf.fit(X_aug_train, Y_aug_train)

# %% [markdown]
'''
# Re-check performance
'''

# %%
Y_test_new_pred = clf.predict(X_test_new)
print('\nClassifier performance -- augmented model on simulated test set')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_new_pred, zero_division=0))

Y_test_pred = clf.predict(X_test)
print('\nClassifier performance -- augmented model on original test set')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_pred, zero_division=0))

# %% [markdown]
'''
Most of the performance loss is recovered with the augmented model.
Also, note how that the performance on the original test set remained almost
unchanged.
'''
