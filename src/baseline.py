# %% [markdown]
'''
# Random forest + temporal models

In this section, we will train a random forest on the extracted windows
from the previous section. We will explore ways to account for the temporal
dependency such as mode smoothing and hidden Markov models.

## Setup
'''

# %%
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import utils

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
# Argument oob_score=True to be used for HMM smoothing (see later below)
clf = BalancedRandomForestClassifier(
    n_estimators=2000,
    replacement=True,
    sampling_strategy='not minority',
    oob_score=True,
    n_jobs=4,
    random_state=42,
    verbose=1
)
clf.fit(X_train, Y_train)

# %% [markdown]
'''
## Model performance
'''
# %%

Y_test_pred = clf.predict(X_test)
print('\nClassifier performance')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_pred))

# %% [markdown]
'''
Overall, the model seems to do well in distinguishing between very inactive
periods ("sit-stand" and "sleep") and very active ones ("bicycling"), but there
seems to be confusion between the remaining activities.

## Plot predicted vs. true activity profiles

Using our utility function, let's plot the activity profile for participant
`P101`.
'''

# %%
mask = pid_test == 'P101'
fig, axs = utils.plot_compare(T_test[mask],
                              Y_test[mask],
                              Y_test_pred[mask],
                              trace=X_test.loc[mask, 'std'])
fig.show()

# %% [markdown]
'''
The profile plots look good at first glance. After all, the majority of
activities happen to be of the sedentary type for which the model performs
well &mdash; this is reflected by the relatively high `weighted avg` scores in
the table report.
However, the `macro avg` scores are still low, and we see that the model
struggles to classify relevant activities such as bicycling and walking.
Moreover, we find some awkward sequences, for example issues with discontinuous
sleep periods. This is because the model is only trained to classify each
window instance independently and does not account for temporal dependencies.

## Accounting for temporal dependency

### Rolling mode smoothing
Let's use rolling mode smoothing to smooth the model predictions: Pick the
most popular label within a rolling time window.

'''

# %%

def mode(alist):
    ''' Mode of a list, but return middle element if ambiguous '''
    m, c = stats.mode(alist)
    m, c = m.item(), c.item()
    if c==1:
        return alist[len(alist)//2]
    return m

def rolling_mode(t, y, window_size='100S'):
    y_dtype_orig = y.dtype
    # Hack to make it work with pandas.Series.rolling()
    y = pd.Series(y, index=t, dtype='category')
    y_code_smooth = y.cat.codes.rolling(window_size).apply(mode, raw=True).astype('int')
    y_smooth = pd.Categorical.from_codes(y_code_smooth, dtype=y.dtype)
    y_smooth = np.asarray(y_smooth, dtype=y_dtype_orig)
    return y_smooth

# %%

# Smooth the predictions of each participant
Y_test_pred_smooth = []
unqP, indP = np.unique(pid_test, return_index=True)
unqP = unqP[np.argsort(indP)]  # keep the order or else we'll scramble our arrays
for p in unqP:
    mask = pid_test == p
    Y_test_pred_smooth.append(rolling_mode(T_test[mask], Y_test_pred[mask]))
Y_test_pred_smooth = np.concatenate(Y_test_pred_smooth)

print('\nClassifier performance -- mode smoothing')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_pred_smooth))

# Check again participant
mask = pid_test == 'P101'
fig, axs = utils.plot_compare(T_test[mask],
                              Y_test[mask],
                              Y_test_pred_smooth[mask],
                              trace=X_test.loc[mask, 'std'])
fig.show()

# %% [markdown]
'''

The simple mode smoothing already improved performance slightly.

### Hidden Markov Model

A more principled approch is to use a Hidden Markov Model (HMM). Here the random
forest predictions are considered as "observations" of the "hidden ground
truth". The emission matrix can be estimated from probabilistic predictions of
model, and the transition matrix can be estimated from the ground truth sequence
of activities. The prior probabilities can be set as the rates observed in the
dataset, or a uniform (uninformative) prior.

Check `utils.train_hmm` and `utils.viterbi` for implementationd details.

'''

# %%

# Use the convenientely provided out-of-bag probability predictions from the
# random forest training process.
# QUESTION: Why not Y_train_prob = clf.predict_proba(X_train) ?
Y_train_prob = clf.oob_decision_function_  # out-of-bag probability predictions
labels = clf.classes_  # need this to know the label order of cols of Y_train_prob
hmm_params = utils.train_hmm(Y_train_prob, Y_train, labels)  # obtain HMM matrices/params
Y_test_pred_hmm = utils.viterbi(Y_test_pred, hmm_params)  # smoothing
print('\nClassifier performance -- HMM smoothing')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_pred_hmm))

# Check again participant
mask = pid_test == 'P101'
fig, ax = utils.plot_compare(T_test[mask],
                             Y_test[mask],
                             Y_test_pred_hmm[mask],
                             trace=X_test.loc[mask, 'std'])
fig.show()

# %% [markdown]
'''
HMM further improves the performance scores.

## Is a simple logistic regression enough?

*Note: this may take a while*
'''

# %%
clf_LR = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    random_state=42)
scaler = StandardScaler()
pipe = make_pipeline(scaler, clf_LR)
pipe.fit(X_train, Y_train)

Y_test_pred_LR = pipe.predict(X_test)

# HMM smoothing
Y_train_LR_prob = pipe.predict_proba(X_train)  # sorry! LR doesn't provide OOB estimates for free
labels = pipe.classes_
hmm_params_LR = utils.train_hmm(Y_train_LR_prob, Y_train, labels)
Y_test_pred_LR_hmm = utils.viterbi(Y_test_pred_LR, hmm_params_LR)  # smoothing

print('\nClassifier performance -- Logistic regression')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_pred_LR_hmm))

# Check again participant
mask = pid_test == 'P101'
fig, axs = utils.plot_compare(T_test[mask],
                      Y_test[mask],
                      Y_test_pred_LR_hmm[mask],
                      trace=X_test.loc[mask, 'std'])
fig.show()

# %% [markdown]
''' The LR model performed well on the easier classes "sleep" and "sit-stand",
but was much worse on all the other classes.
'''
