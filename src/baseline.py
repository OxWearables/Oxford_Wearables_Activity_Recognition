# %% [markdown]
'''
# Random forest + temporal smoothing 

In this section, we will train a random forest on the extracted windows
from the previous section. We will explore ways to account for the temporal
dependency such as mode smoothing and hidden Markov model.

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
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm
import utils  # helper functions -- check out utils.py

# For reproducibility
np.random.seed(42)

# %% [markdown]
''' 
## Load dataset 
'''

# %%

# Path to your extracted windows
DATASET_PATH = 'dataset/'
print(f'Content of {DATASET_PATH}')
print(os.listdir(DATASET_PATH))

with open(DATASET_PATH+'info.json', 'r') as f:
    info = json.load(f)  # load metadata

X = np.memmap(DATASET_PATH+'X.dat', mode='r', dtype=info['X_dtype'], shape=tuple(info['X_shape']))
Y = np.memmap(DATASET_PATH+'Y.dat', mode='r', dtype=info['Y_dtype'], shape=tuple(info['Y_shape']))
T = np.memmap(DATASET_PATH+'T.dat', mode='r', dtype=info['T_dtype'], shape=tuple(info['T_shape']))
P = np.memmap(DATASET_PATH+'P.dat', mode='r', dtype=info['P_dtype'], shape=tuple(info['P_shape']))

print('X shape:', X.shape)
print('Y shape:', Y.shape)
print('T shape:', T.shape)
print('P shape:', P.shape)


# %% [markdown]
'''
## Feature extraction
The feature extraction code used in the previous section can also be found in
*utils.py* file. Feel free to engineer your own features!

*Note: this may take a while*
'''

# %%

# # Extract features
# X_feats = pd.DataFrame([utils.extract_features(x) for x in X])

# # (Optional) Save to disk to avoid recomputation in future runs
# X_feats.to_pickle('X_feats.pkl')

# Reload features
X_feats = pd.read_pickle('X_feats.pkl')

print(X_feats)

# %% [markdown]
'''
Note the great dimensionality reduction, from $3000 \times 3$ to just $22$
values per window &mdash; a reduction of about 400 times.

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
# Argument oob_score=True to be used for HMM smoothing (see below)
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

Y_train_pred = clf.predict(X_train)
Y_test_pred = clf.predict(X_test)
print('\nClassifier performance')
print('In sample:\n', metrics.classification_report(Y_train, Y_train_pred))
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_pred)) 

# %% [markdown]
'''
Overall, the model seems to do well in distinguishing between very inactive
periods ("sit-stand" and "sleep") and very active ones ("bicycling"), but there
seems to be confusion between the remaining activities.

## Plot predicted vs. true activity profiles

Using our utility function, let's plot the activity profile for participant
`006`. Here we also pass the acceleration mean for plotting purposes.
'''

# %%
mask = P_test=='006'
utils.plot_compare_activity(T_test[mask], 
                      Y_test[mask],
                      Y_test_pred[mask], 
                      X_test.loc[mask, 'mean'])


# %% [markdown]
'''
The predictions look good at first glance (after all, the majority of
activities happen to be of the inactive type for which the model performs
well &mdash; this is what the high `weighted avg` in the performance
report reflects). But we find some awkward sequence of activities in the predictions,
for example broken "sleep" patterns insterspersed with "sit-stand"
activities. This is because the model does not account for the temporal
dependency and treats the instances as independent from each other.

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
unqP, indP = np.unique(P_test, return_index=True)
unqP = unqP[np.argsort(indP)]  # keep the order or else we'll scramble our arrays
for p in unqP:
    mask = P_test==p
    Y_test_pred_smooth.append(rolling_mode(T_test[mask], Y_test_pred[mask]))
Y_test_pred_smooth = np.concatenate(Y_test_pred_smooth)

print('\nClassifier performance -- mode smoothing')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_pred_smooth)) 

# Check again participant `006`
mask = P_test=='006'
utils.plot_compare_activity(T_test[mask], 
                      Y_test[mask],
                      Y_test_pred_smooth[mask], 
                      X_test.loc[mask, 'mean'])

# %% [markdown]
'''
### Hidden Markov Model

A more principled approch is to use a Hidden Markov Model (HMM). 
Here the random forest predictions are "observations" of the "hidden ground
truth". The emission matrix can be obtained from probabilistic predictions of
the classifier (`predict_proba()`), and the transition matrix can be obtained
from the ground truth sequence of labels. The prior probabilities of the
labels can be user-specified or set as the label rates observed in the
dataset.

'''

# %%

def train_hmm(Y_prob, Y_true, labels, uninformative_prior=True):
    ''' https://en.wikipedia.org/wiki/Hidden_Markov_model '''

    if uninformative_prior:  
        # All labels with equal probability
        prior = np.ones(len(labels)) / len(labels)
    else:
        # Label probability equals observed rate
        prior = np.mean(Y_true.reshape(-1,1)==labels, axis=0)

    emission = np.vstack(
        [np.mean(Y_prob[Y_true==label], axis=0) for label in labels]
    )
    transition = np.vstack(
        [np.mean(Y_true[1:][(Y_true==label)[:-1]].reshape(-1,1)==labels, axis=0)
            for label in labels]
    )

    params = {'prior':prior, 'emission':emission, 'transition':transition, 'labels':labels}

    return params


def viterbi(Y_obs, hmm_params):
    ''' https://en.wikipedia.org/wiki/Viterbi_algorithm '''

    def log(x):
        SMALL_NUMBER = 1e-16
        return np.log(x + SMALL_NUMBER)

    prior = hmm_params['prior']
    emission = hmm_params['emission']
    transition = hmm_params['transition']
    labels = hmm_params['labels']

    nobs = len(Y_obs)
    nlabels = len(labels)

    Y_obs = np.where(Y_obs.reshape(-1,1)==labels)[1]  # to numeric

    probs = np.zeros((nobs, nlabels))
    probs[0,:] = log(prior) + log(emission[:, Y_obs[0]])
    for j in range(1, nobs):
        for i in range(nlabels):
            probs[j,i] = np.max(
                log(emission[i, Y_obs[j]]) + \
                log(transition[:, i]) + \
                probs[j-1,:])  # probs already in log scale
    viterbi_path = np.zeros_like(Y_obs)
    viterbi_path[-1] = np.argmax(probs[-1,:])
    for j in reversed(range(nobs-1)):
        viterbi_path[j] = np.argmax(
            log(transition[:, viterbi_path[j+1]]) + \
            probs[j,:])  # probs already in log scale

    viterbi_path = labels[viterbi_path]  # to labels

    return viterbi_path

# %%

# Use the convenientely provided out-of-bag probability predictions from the
# random forest training. Question: Why is it preferable over 
# Y_train_prob = clf.predict_proba(X_train)?
Y_train_prob = clf.oob_decision_function_  # out-of-bag probability predictions
labels = clf.classes_  # need this to know the label order of cols of Y_train_prob
hmm_params = train_hmm(Y_train_prob, Y_train, labels)  # obtain HMM matrices/params
Y_test_pred_hmm = viterbi(Y_test_pred, hmm_params)  # smoothing
print('\nClassifier performance -- HMM smoothing')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_pred_hmm)) 

# Check again participant `006`
mask = P_test=='006'
utils.plot_compare_activity(T_test[mask], 
                      Y_test[mask],
                      Y_test_pred_hmm[mask], 
                      X_test.loc[mask, 'mean'])

# %% [markdown]
'''
HMM further improves the performance overall.

## Is a simple logistic regression enough?

*Note: this may take a while*
'''

# %%
clf_LR = LogisticRegression(
    max_iter=1000, 
    multi_class='multinomial', 
    random_state=42, 
    verbose=1)
scaler = StandardScaler()
pipe = make_pipeline(scaler, clf_LR)
pipe.fit(X_train, Y_train)

Y_test_pred_LR = pipe.predict(X_test)

# HMM smoothing
Y_train_LR_prob = pipe.predict_proba(X_train)  # sorry! LR doesn't provide OOB estimates for free
labels = pipe.classes_
hmm_params_LR = train_hmm(Y_train_LR_prob, Y_train, labels)
Y_test_pred_LR_hmm = viterbi(Y_test_pred_LR, hmm_params_LR)  # smoothing

print('\nClassifier performance -- Logistic regression')
print('Out of sample:\n', metrics.classification_report(Y_test, Y_test_pred_LR_hmm)) 

# Check again participant `006`
mask = P_test=='006'
utils.plot_compare_activity(T_test[mask], 
                      Y_test[mask],
                      Y_test_pred_LR_hmm[mask], 
                      X_test.loc[mask, 'mean'])

# %% [markdown]
''' The model performs well on the easier classes "sleep" and "sit-stand",
but is much worse on most of the remaining ones.
'''


# %%
