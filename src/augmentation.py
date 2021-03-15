# %% [markdown] 
'''
# Data augmentation

Data augmentation is a straighforward way to artificially increase the size
of the dataset by applying transformations to the data. It is very useful
when the dataset is small, or to reduce overfitting in large models.
But not all transformations are applicable. For example, for image
recognition it may not matter if the image is rotated, flipped or stretched.
On the other hand, flipping and rotating may not be a good idea for
handwriting recognition (e.g. a "p" gets turned into a "q" or "b", a "6" into
a "9"). The key is to find *invariances* of the *learning task*.

## Setup
'''
# %%
import os
import json
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import metrics
from tqdm.auto import tqdm
import utils  # helper functions -- check out utils.py

# For reproducibility
np.random.seed(42)

# %% [markdown]
''' 
## Load dataset 
'''

# %%

DATASET_PATH = 'dataset/'  # path to your extracted windows
X_FEATS_PATH = 'X_feats.pkl'  # path to your extracted features
print(f'Content of {DATASET_PATH}')
print(os.listdir(DATASET_PATH))

with open(DATASET_PATH+'info.json', 'r') as f:
    info = json.load(f)  # load metadata

X = np.memmap(DATASET_PATH+'X.dat', mode='r', dtype=info['X_dtype'], shape=tuple(info['X_shape']))
Y = np.memmap(DATASET_PATH+'Y.dat', mode='r', dtype=info['Y_dtype'], shape=tuple(info['Y_shape']))
T = np.memmap(DATASET_PATH+'T.dat', mode='r', dtype=info['T_dtype'], shape=tuple(info['T_shape']))
P = np.memmap(DATASET_PATH+'P.dat', mode='r', dtype=info['P_dtype'], shape=tuple(info['P_shape']))
X_feats = pd.read_pickle('X_feats.pkl')

print('X shape:', X.shape)
print('Y shape:', Y.shape)
print('T shape:', T.shape)
print('P shape:', P.shape)
print('X_feats shape:', X_feats.shape)

# %% [markdown]
'''
## Train/test split
'''

# %%

# Take out 10 participants
test_ids = ['002', '003', '004', '005', '006', 
            '007', '008', '009', '010', '011']
mask_test = np.isin(P, test_ids)
mask_train = ~mask_test
X_train, Y_train, P_train, T_train = \
    X_feats[mask_train], Y[mask_train], P[mask_train], T[mask_train]
X_test, Y_test, P_test, T_test = \
    X_feats[mask_test], Y[mask_test], P[mask_test], T[mask_test]
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
training set were right-handed, but the test subjects are left-handed.
This would more or less result in the device being rotated.

<img src="wrist_accelerometer.jpg" width="200"/>

Let's generate an artificial test set simulating this scenario by flipping
two of the axes signs (it does not exactly simulate a rotation since the
movement dynamics are also mirrored, but it is enough for our demonstration
purposes).

*Note: this may take a while*
'''

# %%
# Split raw data into train and test set
# X[mask_train] and X[mask_test] if you like to live dangerously
X_raw_train = utils.ArrayFromMask(X, mask_train)
X_raw_test = utils.ArrayFromMask(X, mask_test)

print("Creating test set with 'rotated device'...")
X_rot_test = []
for i in tqdm(range(X_raw_test.shape[0])):
    # Rotate device
    x = X_raw_test[i].copy()
    x[:,1] *= -1
    x[:,2] *= -1
    X_rot_test.append(utils.extract_features(x))
X_rot_test = pd.DataFrame(X_rot_test)

# %% [markdown]
''' ### Performance on simulated test set '''

# %%

Y_rot_test_pred = clf.predict(X_rot_test)
print('\nClassifier performance -- simulated test set')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_rot_test_pred, zero_division=0)) 

# %% [markdown]
'''
The model performance is notably worse on the simulated set. The solution is
simply to augment the training dataset with the rotation &mdash; in
general, with transformations that we wish our model to be invariant to. In
our case, we wish our model to perform well no matter how the device was
worn.

## Data augmentation

*Note: this may take a while*
'''

# %%
print("Creating training set with 'rotated device'...")
X_rot_train = []
for i in tqdm(range(X_raw_train.shape[0])):
    # Rotate device
    x = X_raw_train[i].copy()
    x[:,1] *= -1
    x[:,2] *= -1
    X_rot_train.append(utils.extract_features(x))
X_rot_train = pd.DataFrame(X_rot_train)

# Add the "new data" to training set
X_aug_train = pd.concat((X_train, X_rot_train))
Y_aug_train = np.concatenate((Y_train, Y_train))
print("X_aug_train shape:", X_aug_train.shape)

# %% [markdown]
''' ### Re-train and check performance

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

Y_rot_test_pred = clf.predict(X_rot_test)
print('\nClassifier performance -- augmented model on simulated test set')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_rot_test_pred, zero_division=0)) 

# %% [markdown]
'''
Some of the performance is recovered with the augmented model.
'''
