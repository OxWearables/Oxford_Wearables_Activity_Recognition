# %%
'''
# Activity recognition on the Capture24 dataset

## Semi-supervised learning

While digital data collection is becoming easier and cheaper, labeling such data
still requires expensive and time-consuming human labor.
For example, while it is possible to label accelerometer measurements for ~150
participants as in our Capture24 dataset, it is unfeasible to do so for the
tens of thousands of *unlabeled* accelerometer measurements that are
currently available in the [UK
Biobank](https://www.ukbiobank.ac.uk/activity-monitor-3/) because *a)*
compliance to wear body cameras is much lower than wrist-worn accelerometers
and *b)* the human labor to go through all the camera recordings would be
very expensive. Semi-supervised learning is therefore of great interest,
where the aim is to somehow use the unlabeled data to improve the model
performance.

###### Setup
'''

# %%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
import utils

# For reproducibility
np.random.seed(42)

# %%
''' ###### Load dataset and hold out some instances for testing '''

# %%
data = np.load('capture24.npz', allow_pickle=True)
# data = np.load('capture24_small.npz', allow_pickle=True)
print("Contents of capture24.npz:", data.files)
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
## Self-training

One of the simplest semi-supervised methods is based on proxy-labels via self-training. The idea is to simply evaluate a trained model on the unlabeled instances and incorporate those with high confidence predictions into the training set, then re-train the model on the augmented set. This process is repeated several times until some criteria is met, e.g. when no more instances are being included in the training set.
This simple technique works well when the initial model is already very
strong. If the initial model is weak, however, it may reinforce the mistakes
in its predictions.
In the following, we first train a random forest on the labelled training
set, then evaluate the model on the provided unlabelled dataset
`capture24_test.npz` for self-training.
'''

# %%
# initial model
classifier = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4)
classifier.fit(X_train, y_train)

# Load unlabelled dataset for self-training
data_unl = np.load('capture24_test.npz')
print("\nContents of capture24_test.npz:", data_unl.files)
X_unl = data_unl['X_feats']
print("Shape of X_unl:", X_unl.shape)

# %%
'''
###### Self-training

*Note: this takes several minutes*
'''

# %%
# initial predictions and self-training parameters
y_unl_pred = classifier.predict(X_unl)
y_unl_prob = classifier.predict_proba(X_unl)
y_unl_pred_old = None
max_iter = 5
prob_threshold = 0.8

for i in tqdm(range(max_iter)):

    if np.array_equal(y_unl_pred, y_unl_pred_old):
        tqdm.write("Iteration stopped: no more change found in self-training")
        break

    y_unl_pred_old = np.copy(y_unl_pred)
    confident_mask = np.any(y_unl_prob > prob_threshold, axis=1)
    tqdm.write(f"Using {np.sum(confident_mask)} instances from the unlabeled set")

    # re-train on augmented set
    classifier.fit(
        np.vstack((X_train, X_unl[confident_mask])),
        np.hstack((y_train, y_unl_pred_old[confident_mask]))
    )

    # updated predictions
    y_unl_pred = classifier.predict(X_unl)
    y_unl_prob = classifier.predict_proba(X_unl)

# %%
''' ###### Smooth the predictions via HMM and evaluate '''

# %%
Y_oob = classifier.oob_decision_function_[:y_train.shape[0]]
prior, emission, transition = utils.train_hmm(Y_oob, y_train)
y_test_pred = classifier.predict(X_test)
y_test_hmm = utils.viterbi(y_test_pred, prior, transition, emission)
print("\n--- Random forest performance with self-training and HMM smoothing ---")
utils.print_scores(utils.compute_scores(y_test, y_test_hmm))

# %%
'''
###### Ideas

- Tune acceptance threshold of high confidence predictions.
- Incorporate the HMM smoothing into the self-training loop.

###### References

- [A nice summary of proxy-labels methods](https://ruder.io/semi-supervised/)
- [Semi-supervised methods in sklearn](https://scikit-learn.org/stable/modules/label_propagation.html)
'''
