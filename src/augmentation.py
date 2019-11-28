# %%
'''
# Activity recognition on the Capture24 dataset

## Data augmentation

Data augmentation is a straighforward way to artificially increase the size
of the dataset, which can be useful when the dataset is small or when
using large models such as deep neural networks. It relies on generating
variations of each instance in the dataset by applying several transformations on it.

Not any transformation is applicable: For example, for image recognition it may not matter if the image is rotated, flipped or stretched; but for digits and letters recognition, flipping the image may not be helpful.
The key is to find *invariances* in the data that are applicable for our learning task.

###### Setup
'''

# %%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm
import utils  # contains helper functions for this workshop -- check utils.py

# For reproducibility
np.random.seed(42)

# A function to plot the activity timeseries of a participant
def plot_activity(X, y, pid, time, ipid=3):
    mask = pid == ipid
    # The first hand-crafted feature X[:,0] is mean acceleration
    return utils.plot_activity(X[:,0][mask], y[mask], time[mask])

# %%
''' ###### Load dataset and hold out some instances for testing

To highlight the utility of data augmentation in small datasets, let us constrain ourselves to only *five* participants:
'''

# %%
data = np.load('capture24.npz', allow_pickle=True)
# data = np.load('capture24_small.npz', allow_pickle=True)
mask = np.isin(data['pid'], [1, 2, 3, 4, 5])  # take only five participants
X_feats, y, pid, time = \
    data['X_feats'][mask], data['y'][mask], data['pid'][mask], data['time'][mask]
print("Contents of capture24.npz:", data.files)

# Hold out some participants for testing the model
pids_test = [2, 3]  # participants 2 & 3
mask_test = np.isin(pid, pids_test)
mask_train = ~mask_test
X_train, y_train, pid_train, time_train = \
    X_feats[mask_train], y[mask_train], pid[mask_train], time[mask_train]
X_test, y_test, pid_test, time_test = \
    X_feats[mask_test], y[mask_test], pid[mask_test], time[mask_test]
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# %%
''' ###### Baseline

Train the random forest and evaluate on the held out participants

*Note: this takes a few minutes*
'''

# %%
# Training
classifier = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4, verbose=True)
classifier.fit(X_train, y_train)
Y_oob = classifier.oob_decision_function_
prior, emission, transition = utils.train_hmm(Y_oob, y_train)

# Testing
y_test_pred = utils.viterbi(classifier.predict(X_test), prior, transition, emission)
print("\n--- Baseline performance ---")
print("Cohen kappa score:", utils.cohen_kappa_score(y_test, y_test_pred, pid_test))
print("Accuracy score:", utils.accuracy_score(y_test, y_test_pred, pid_test))

# %%
'''
## Robustness to unforseen circumstances

What would our performance be if our two left-out participants happened to wear the device with a different orientation?
For example: Suppose that all our training participants wore the device on their right hand, then how would our model perform on participants that wore the device on their left hand?
This scenario corresponds to a 180 degrees rotation around the z-axis:
<img src="ax3_orientation.jpg" width="400"/>
Let's generate a pseudo-test set simulating this scenario: '''

# %%
# First load the raw triaxial data to perform the rotation on it
X_raw = np.load('X_raw.npy', mmap_mode='r')
# X_raw = np.load('X_raw_small.npy')
# X_raw[mask_train] and X_raw[mask_test] if you like to live dangerously
X_raw = utils.ArrayFromMask(X_raw, mask)  # grab the five participants
X_raw_train = utils.ArrayFromMask(X_raw, mask_train)
X_raw_test = utils.ArrayFromMask(X_raw, mask_test)

# Initialize feature extractor -- this needs to be done only once
extractor = utils.Extractor()

print("Extracting features on pseudo-test set...")
X_test_rot = np.empty_like(X_test)
y_test_rot = y_test.copy()
for i in tqdm(range(X_raw_test.shape[0])):
    # Rotate instance around z-axis and extract features
    x = X_raw_test[i].copy()
    x[0,:] *= -1
    x[1,:] *= -1
    X_test_rot[i] = extractor.extract(x)

# %%
''' ###### How does the baseline model perform on the pseudo-set? '''

# %%
y_test_rot_pred = utils.viterbi(
    classifier.predict(X_test_rot), prior, transition, emission)
print("\n--- Performance of baseline model on pseudo-test set ---")
print("Cohen kappa score:", utils.cohen_kappa_score(y_test_rot, y_test_rot_pred, pid_test))
print("Accuracy score:", utils.accuracy_score(y_test_rot, y_test_rot_pred, pid_test))

# %%
'''
The model score has dropped drastically.
Let's visualize the predicted activities for participant #3 and the pseudo-participant #3 (with rotated device):
'''

# %%
fig, _ = plot_activity(X_test, y_test_pred, pid_test, time_test, ipid=3)
fig.suptitle('participant #3', fontsize='small')
fig.show()

fig, _ = plot_activity(X_test_rot, y_test_rot_pred, pid_test, time_test, ipid=3)
fig.suptitle('pseudo-participant #3', fontsize='small')
fig.show()

# %%
'''
As we see, the activity plot has changed significantly &mdash; the model is not
robust to participants wearing the device differently.
Ideally, we would like our model to perform well regardless of
how the device was worn.

###### Data augmentation

We can incorporate the desired invariance by simply augmenting our training set and re-training the model:
'''

# %%
print("\nExtracting features on pseudo-training set...")
X_train_rot = np.empty_like(X_train)
y_train_rot = y_train.copy()
for i in tqdm(range(X_raw_train.shape[0])):
    # Rotate instance around z-axis and extract features
    x = X_raw_train[i].copy()
    x[0,:] *= -1
    x[1,:] *= -1
    X_train_rot[i] = extractor.extract(x)

# Add in the "new data" to training set
X_train = np.concatenate((X_train, X_train_rot))
y_train = np.concatenate((y_train, y_train_rot))
print("Shape of new augmented X_train:", X_train.shape)

# %%
''' ###### Re-train the model on the augmented training set

*Note: this takes a few minutes*
'''

# %%
classifier = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4, verbose=True)
classifier.fit(X_train, y_train)
Y_oob = classifier.oob_decision_function_
prior, emission, transition = utils.train_hmm(Y_oob, y_train)

# %%
''' ###### Re-evaluate the model '''

# %%
print("\n--- Performance of re-trained model on the original test set ---")
y_test_pred = utils.viterbi(classifier.predict(X_test), prior, transition, emission)
print("Cohen kappa score:", utils.cohen_kappa_score(y_test, y_test_pred, pid_test))
print("Accuracy score:", utils.accuracy_score(y_test, y_test_pred, pid_test))

print("\n--- Performance of re-trained model on the pseudo-test set ---")
y_test_rot_pred = utils.viterbi(classifier.predict(X_test_rot), prior, transition, emission)
print("Cohen kappa score:", utils.cohen_kappa_score(y_test_rot, y_test_rot_pred, pid_test))
print("Accuracy score:", utils.accuracy_score(y_test_rot, y_test_rot_pred, pid_test))

fig, _ = plot_activity(X_test, y_test_pred, pid_test, time_test, ipid=3)
fig.suptitle('participant #3', fontsize='small')
fig.show()

fig, _ = plot_activity(X_test_rot, y_test_rot_pred, pid_test, time_test, ipid=3)
fig.suptitle('pseudo-participant #3', fontsize='small')
fig.show()

# %%
'''
As we see, by data-augmenting the training set with the desired invariance
the model prediction is more robust to wear variations.

###### Ideas

- What other invariances should we want our model to learn?
- Re-run the notebook on the whole dataset. Can you explain the reduced discrepancy?
'''
