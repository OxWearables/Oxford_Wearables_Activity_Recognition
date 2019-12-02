import numpy as np
import sklearn.metrics as metrics
import tempfile
# from numba import jit


# class code (tip to remember: goes from lower to higher activity)
CLASS_CODE = {'sleep': 0, 'sedentary': 1, 'tasks-light': 2, 'walking': 3, 'moderate': 4}
# list of classes, ordered by code
CLASSES = ['sleep', 'sedentary', 'tasks-light', 'walking', 'moderate']
NUM_CLASSES = len(CLASSES)
# colors to be used for each class, ordered by code
COLORS = ['blue', 'red', 'darkorange', 'lightgreen', 'green']

NUM_FEATS = 125  # number of hand-crafted features
SAMPLE_RATE = 100  # device sample rate (100hz)
RAW_SHAPE = (3,3000)  # triaxial x 30secs x 100hz
RAW_DTYPE = 'float32'  # measurements are in float32 precision


def load_raw(filepath):
    return np.memmap(filepath, dtype=RAW_DTYPE, mode='r').reshape(-1,*RAW_SHAPE)


def encode_one_hot(y):
    '''
    0 -> 1,0,0,0,0
    1 -> 0,1,0,0,0
    2 -> 0,0,1,0,0
    3 -> 0,0,0,1,0
    4 -> 0,0,0,0,1
    '''
    return (y.reshape(-1,1) == np.arange(NUM_CLASSES)).astype(int)


def train_hmm(Y_pred, y_true):
    ''' https://en.wikipedia.org/wiki/Hidden_Markov_model '''
    if Y_pred.ndim == 1 or Y_pred.shape[1] == 1:
        Y_pred = encode_one_hot(Y_pred)

    prior = np.mean(y_true.reshape(-1,1) == np.arange(NUM_CLASSES), axis=0)
    emission = np.vstack(
        [np.mean(Y_pred[y_true==i], axis=0) for i in range(NUM_CLASSES)]
    )
    transition = np.vstack(
        [np.mean(y_true[1:][(y_true==i)[:-1]].reshape(-1,1) == np.arange(NUM_CLASSES), axis=0)
            for i in range(NUM_CLASSES)]
    )
    return prior, emission, transition


def viterbi(y_pred, prior, transition, emission):
    ''' https://en.wikipedia.org/wiki/Viterbi_algorithm '''
    small_number = 1e-16

    def log(x):
        return np.log(x + small_number)

    num_obs = len(y_pred)
    probs = np.zeros((num_obs, NUM_CLASSES))
    probs[0,:] = log(prior) + log(emission[:, y_pred[0]])
    for j in range(1, num_obs):
        for i in range(NUM_CLASSES):
            probs[j,i] = np.max(
                log(emission[i, y_pred[j]]) + \
                log(transition[:, i]) + \
                probs[j-1,:])  # probs already in log scale
    viterbi_path = np.zeros_like(y_pred)
    viterbi_path[-1] = np.argmax(probs[-1,:])
    for j in reversed(range(num_obs-1)):
        viterbi_path[j] = np.argmax(
            log(transition[:, viterbi_path[j+1]]) + \
            probs[j,:])  # probs already in log scale

    return viterbi_path


def compute_scores(y_true, y_pred):
    ''' Compute a bunch of scoring functions '''
    confusion = metrics.confusion_matrix(y_true, y_pred)
    per_class_recall = metrics.recall_score(y_true, y_pred, average=None)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    balanced_acuracy = metrics.balanced_accuracy_score(y_true, y_pred)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    return {
        'confusion':confusion,
        'per_class_recall':per_class_recall,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acuracy,
        'kappa':kappa,
    }


def print_scores(scores):
    print("Accuracy score:", scores['accuracy'])
    print("Balanced accuracy score:", scores['balanced_accuracy'])
    print("Cohen kappa score:", scores['kappa'])
    print("\nPer-class recall scores:")
    print(
        "sleep      : {}\n"
        "sedentary  : {}\n"
        "tasks-light: {}\n"
        "walking    : {}\n"
        "moderate   : {}".format(*scores['per_class_recall'])
    )
    print("\nConfusion matrix:\n", scores['confusion'])


def cohen_kappa_score(y_true, y_pred, pid=None):
    """ Compute kappa score accounting for groups (given by pid) """
    if pid is None:
        return metrics.cohen_kappa_score(y_true, y_pred)
    else:
        kappas = []
        for i in np.unique(pid):
            _y_true = y_true[pid == i]
            _y_pred = y_pred[pid == i]
            if len(np.unique(_y_true)) <= 1:
                # cannot compute kappa with only one class
                continue
            kappas.append(metrics.cohen_kappa_score(_y_true, _y_pred))
        return np.mean(kappas)


def accuracy_score(y_true, y_pred, pid=None):
    """ Compute accuracy score accounting for groups (given by pid) """
    if pid is None:
        return metrics.accuracy_score(y_true, y_pred)
    else:
        accuracys = []
        for i in np.unique(pid):
            _y_true = y_true[pid == i]
            _y_pred = y_pred[pid == i]
            accuracys.append(metrics.accuracy_score(_y_true, _y_pred))
        return np.mean(accuracys)


# @jit(nopython=True)
def sum_sqrsum(alist):
    ''' Compute sum and sum of squares. Intended for large memmap arrays that
    do not fit in memory. It uses Kahan summation algorithm to reduce rounding errors:
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm '''
    asum = np.zeros_like(alist[0])
    asqrsum = np.zeros_like(alist[0])

    c_sum = np.zeros_like(alist[0])
    c_sqrsum = np.zeros_like(alist[0])

    for i in range(len(alist)):

        a_sum = alist[i]
        y_sum = a_sum - c_sum
        t_sum = asum + y_sum
        c_sum = (t_sum - asum) - y_sum
        asum = t_sum

        a_sqrsum = alist[i]**2
        y_sqrsum = a_sqrsum - c_sqrsum
        t_sqrsum = asqrsum + y_sqrsum
        c_sqrsum = (t_sqrsum - asqrsum) - y_sqrsum
        asqrsum = t_sqrsum

    return asum, asqrsum


def mu_std(alist):
    ''' Mean and standad deviation of a list of numbers or arrays '''
    asum, asqrsum = sum_sqrsum(alist)
    mu = asum / len(alist)
    var = np.maximum(0, asqrsum/len(alist) - mu**2)
    std = np.sqrt(var)
    return mu, std


class ArrayFromIdxs(object):
    ''' X[idxs] without returning a copy. Intended for
    large memmap arrays that do not fit in memory.'''
    def __init__(self, X, idxs):
        self.X = X
        self.idxs = np.asarray(idxs)
        self.shape = (len(idxs), *X.shape[1:])
        self.dtype = X.dtype

    def __getitem__(self, item):
        return self.X[self.idxs[item]]

    def __len__(self):
        return len(self.idxs)


class ArrayFromMask(ArrayFromIdxs):
    ''' X[mask] without returning a copy. Intended for
    large memmap arrays that do not fit in memory.'''
    def __init__(self, X, mask):
        idxs = np.where(mask)[0]
        super().__init__(X, idxs)


def memmap_from_idxs(X, idxs):
    ''' Create a memmap for X[idxs] on a temporary file '''
    idxs = sorted(set(idxs))
    dtype = X.dtype
    shape = (len(idxs), *X.shape[1:])
    X_new = np.memmap(tempfile.TemporaryFile(), mode='w+', dtype=dtype, shape=shape)
    NFLUSH = 1048
    for i, idx in enumerate(idxs):
        X_new[i] = X[idx]
        if (i+1) % NFLUSH ==0 or i == (len(idxs)-1):
            X_new.flush()
    return X_new


def memmap_from_mask(X, mask):
    ''' Create a memmap for X[mask] on a temporary file '''
    idxs = np.where(mask)[0]
    return memmap_from_idxs(X, idxs)


# -------------------------------------------
#  Code for hand-crafted feature extraction
# -------------------------------------------
import jpype
import jpype.imports
class Extractor():
    ''' A wrapper of the Java class FeatureExtractor using the JPype package.
    It starts a Java Virtual Machine, instantiates FeatureExtractor, and
    implements 'extract' method to handle numpy arrays.
    '''
    def __init__(self):
        # start Java Virtual Machine and instantiate FeatureExtractor
        if not jpype.isJVMStarted():
            jpype.addClassPath("java_feature_extractor")
            jpype.addClassPath("java_feature_extractor/JTransforms-3.1-with-dependencies.jar")
            jpype.startJVM(convertStrings=False)
        self.java_extractor = jpype.JClass('FeatureExtractor')

    def extract(self, xyz):
        xArray, yArray, zArray = xyz
        xArray = jpype.JArray(jpype.JFloat, 1)(xArray)
        yArray = jpype.JArray(jpype.JFloat, 1)(yArray)
        zArray = jpype.JArray(jpype.JFloat, 1)(zArray)
        return np.asarray(self.java_extractor.extract(
            xArray, yArray, zArray, SAMPLE_RATE))


# ----------------------------------------
#  Function for activity timeseries plot
# ----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime, timedelta, time
def plot_activity(x, y, t):
    ''' Plot activity timeseries '''
    BACKGROUND_COLOR = '#d3d3d3' # lightgray

    def split_by_timegap(group, seconds=30):
        subgroupIDs = (group.index.to_series().diff() > timedelta(seconds=seconds)).cumsum()
        subgroups = group.groupby(by=subgroupIDs)
        return subgroups

    convert_date = np.vectorize(
        lambda day, x: matplotlib.dates.date2num(datetime.combine(day, x)))
    timeseries = pd.DataFrame(data={'x':x, 'y':y, 't':t})
    timeseries.set_index('t', inplace=True)
    timeseries['x'] = timeseries['x'].rolling(window=12, min_periods=1).mean()  #! inplace?
    ylim_min, ylim_max = np.min(x), np.max(x)
    groups = timeseries.groupby(timeseries.index.date)
    fig, axs = plt.subplots(nrows=len(groups) + 1)
    for ax, (day, group) in zip(axs, groups):
        for _, subgroup in split_by_timegap(group):
            _t = convert_date(day, subgroup.index.time)
            _ys = [(subgroup['y'] == i).astype('int') * ylim_max for i in range(NUM_CLASSES)]
            ax.plot(_t, subgroup['x'], c='k')
            ax.stackplot(_t, _ys, colors=COLORS, alpha=.5, edgecolor='none')

        ax.get_xaxis().grid(True, which='major', color='grey', alpha=0.5)
        ax.get_xaxis().grid(True, which='minor', color='grey', alpha=0.25)
        ax.set_xlim((datetime.combine(day,time(0, 0, 0, 0)),
            datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0))))
        ax.set_xticks(pd.date_range(start=datetime.combine(day,time(0, 0, 0, 0)),
            end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
            freq='4H'))
        ax.set_xticks(pd.date_range(start=datetime.combine(day,time(0, 0, 0, 0)),
            end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
            freq='1H'), minor=True)
        ax.set_ylim((ylim_min, ylim_max))
        ax.get_yaxis().set_ticks([]) # hide y-axis lables
        ax.spines['top'].set_color(BACKGROUND_COLOR)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor(BACKGROUND_COLOR)

        ax.set_title(
            day.strftime("%A,\n%d %B"), weight='bold',
            x=-.2, y=.5,
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation='horizontal',
            transform=ax.transAxes,
            fontsize='medium',
            color='k'
        )

    # legends
    axs[-1].axis('off')
    legend_patches = []
    legend_patches.append(mlines.Line2D([], [], color='k', label='acceleration'))
    for color, label in zip(COLORS, CLASSES):
        legend_patches.append(mpatches.Patch(facecolor=color, label=label, alpha=0.5))
    axs[-1].legend(handles=legend_patches, bbox_to_anchor=(0., 0., 1., 1.),
        loc='center', ncol=min(3,len(legend_patches)), mode="best",
        borderaxespad=0, framealpha=0.6, frameon=True, fancybox=True)
    axs[-1].spines['left'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['bottom'].set_visible(False)

    # format x-axis to show hours
    fig.autofmt_xdate()
    hours = [(str(hr) + 'am') if hr<=12 else (str(hr-12) + 'pm') for hr in range(0,24,4)]
    axs[0].set_xticklabels(hours)
    axs[0].tick_params(labelbottom=False, labeltop=True, labelleft=False)

    return fig, axs
