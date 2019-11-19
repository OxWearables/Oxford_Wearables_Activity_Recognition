# %%
'''
# Activity recognition on the Capture24 dataset

## Semi-supervised learning

While digital data collection is becoming easier and cheaper, labeling such data
still requires expensive and time-consuming human labor.
For example, while it is possible to label accelerometer readings for ~150
participants as in our Capture24 dataset, it is unfeasible to do so for the
the tens of thousands of *unlabeled* accelerometer measurements that are currently available in the [UK Biobank](https://en.wikipedia.org/wiki/UK_Biobank), since *a)* compliance to wear a body camera
is much lower than a wrist-worn accelerometer and *b)* the human labor to go
through all the camera recordings would be very expensive.
Semi-supervised learning is therefore of great interest, where the aim is to somehow use
the unlabeled data to improve the model performance.

###### Setup
'''

# %%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# from tqdm import tqdm
from tqdm.notebook import tqdm
import utils

# For reproducibility
np.random.seed(42)

# %%
''' ###### Load dataset and hold out some instances for testing '''

# %%
# data = np.load('capture24.npz', allow_pickle=True)
data = np.load('capture24_small.npz', allow_pickle=True)
print("Contents of:", data.files)
X, y, pid, time = data['X_feats'], data['y'], data['pid'], data['time']

# Hold out some participants for testing the model
test_pids = [2, 3]
test_mask = np.isin(pid, test_pids)
train_mask = ~np.isin(pid, test_pids)
X_train, y_train, pid_train = X[train_mask], y[train_mask], pid[train_mask]
X_test, y_test, pid_test = X[test_mask], y[test_mask], pid[test_mask]

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)


# %%
'''
One of the simplest semi-supervised methods is using proxy-labels via self-training. The idea is to simply evaluate a trained model on the unlabeled instances and incorporate those with high confidence predictions into the training set, then re-train the model on the augmented set. This process is repeated several times until some criteria is met, e.g. when no more instances are being included in the training set.
This simple technique works well when the initial model is already very strong. If the initial model is weak, however, it may reinforce the mistakes in its predictions.

In the following, we train a random forest classifier with self-training:
'''

# %%
classifier = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=2)

# initial model and predictions
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
y_test_prob = classifier.predict_proba(X_test)
y_test_pred_old = None
max_iter = 10
prob_threshold = 0.8

for i in tqdm(range(max_iter)):

    if np.array_equal(y_test_pred, y_test_pred_old):
        tqdm.write("Iteration stopped: no more change found in self-training")
        break

    y_test_pred_old = np.copy(y_test_pred)
    confident_mask = np.any(y_test_prob > prob_threshold, axis=1)
    tqdm.write(f"Using {np.sum(confident_mask)} instances from the test set")

    # re-train on augmented set
    classifier.fit(
        np.vstack((X_train, X_test[confident_mask])),
        np.hstack((y_train, y_test_pred_old[confident_mask]))
    )

    # updated predictions
    y_test_pred = classifier.predict(X_test)
    y_test_prob = classifier.predict_proba(X_test)

# %%
''' Smooth the predictions via HMM and evaluate: '''

# %%
Y_oob = classifier.oob_decision_function_[:y_train.shape[0]]
prior, emission, transition = utils.train_hmm(Y_oob, y_train)
y_test_pred = classifier.predict(X_test)
y_test_hmm = utils.viterbi(y_test_pred, prior, transition, emission)
print("\n--- Random forest performance with self-training and HMM smoothing ---")
print("Cohen kappa score:", utils.cohen_kappa_score(y_test, y_test_hmm, pid_test))
print("Accuracy score:", utils.accuracy_score(y_test, y_test_hmm, pid_test))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_hmm))

# %%
'''
###### Ideas

- Incorporate the HMM smoothing into the self-training loop.

###### References

- [A nice summary of proxy-labels methods](https://ruder.io/semi-supervised/)
- [Semi-supervised methods in sklearn](https://scikit-learn.org/stable/modules/label_propagation.html)
'''
