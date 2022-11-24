"""
Perform inference with pretrained model on a UKB accelerometer file.

Arguments:
    Input file, in the format .cwa.gz

Example usage:
    python template.py sample.cwa.gz

Output:
    Prediction DataFrame in {eid}.csv format, stored in OUTPUT_PATH
    If the input file is stored in a groupX folder, output will be in OUTPUT_PATH/groupX/
    A {eid}_summary.csv file with the calculated phenotype: 1 row, an 'eid' column followed by the phenotype column(s)
    An {eid}_info.csv file will also be saved with the actipy info dict.
"""

import actipy
import argparse
import os
import torch
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

SAMPLE_RATE = 30  # Hz
WINDOW_SEC = 30  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(SAMPLE_RATE * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(SAMPLE_RATE * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks

start_time = datetime.now()

OUTPUT_PATH = './output'


def vectorized_stride_v2(acc, time, window_size, stride_size):
    """
    Numpy vectorised windowing with stride (super fast!). Will discard the last window.

    :param np.ndarray acc: Accelerometer data array, shape (nsamples, nchannels)
    :param np.ndarray time: Time array, shape (nsamples, )
    :param int window_size: Window size in n samples
    :param int stride_size: Stride size in n samples
    :return: Windowed data and time arrays
    :rtype: (np.ndarray, np.ndarray)
    """
    start = 0
    max_time = len(time)

    sub_windows = (start +
                   np.expand_dims(np.arange(window_size), 0) +
                   # Create a rightmost vector as [0, V, 2V, ...].
                   np.expand_dims(np.arange(max_time + 1, step=stride_size), 0).T
                   )[:-1]  # drop the last one

    return acc[sub_windows], time[sub_windows]


def df_to_windows(df):
    """
    Convert a time series dataframe (e.g.: from actipy) to a windowed Numpy array.

    :param pd.DataFrame df: A dataframe with DatetimeIndex and x, y, z columns
    :return: Data array with shape (nwindows, WINDOW_LEN, 3), Time array with shape (nwindows, )
    :rtype: (np.ndarray, np.ndarray)
    """

    acc = df[['x', 'y', 'z']].to_numpy()
    time = df.index.to_numpy()

    # convert to windows
    x, t = vectorized_stride_v2(acc, time, WINDOW_LEN, WINDOW_STEP_LEN)

    # drop the whole window if it contains a NaN
    na = np.isnan(x).any(axis=1).any(axis=1)
    x = x[~na]
    t = t[~na]

    return x, t[:, 0]  # only return the first timestamp for each window


def extract_features(xyz):
    """ Extract features. xyz is an array of shape (N,3) """

    feats = {}
    feats['xMean'], feats['yMean'], feats['zMean'] = np.mean(xyz, axis=0)
    feats['xStd'], feats['yStd'], feats['zStd'] = np.std(xyz, axis=0)
    v = np.linalg.norm(xyz, axis=1)  # magnitude stream
    feats['mean'], feats['std'] = np.mean(v), np.std(v)

    return feats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='SSL UKB', usage='Apply the model on a UKB cwa file.')
    parser.add_argument('input_file', type=str, help='input cwa file')
    args = parser.parse_args()

    input_file = args.input_file
    input_path = Path(input_file)

    # get person id (pid) and group from input string
    pid = input_path.stem.split('_')[0]
    group = input_path.parent.stem if 'group' in input_path.parent.stem else None

    print(input_file)
    print(group, pid)

    np.random.seed(42)
    torch.manual_seed(42)

    # load data
    data, info = actipy.read_device(input_file,
                                    lowpass_hz=None,
                                    calibrate_gravity=True,
                                    detect_nonwear=True,
                                    resample_hz=SAMPLE_RATE)
    print(data.head(1))
    print(info)
    info = pd.DataFrame(info, index=[0])

    # store original start/end times for reindexing later
    data_start = data.index[0]
    data_end = data.index[-1]

    # prepare dataset
    print('Windowing')
    X, T = df_to_windows(data)

    print(f'X shape: {X.shape}')
    print(f'T shape: {T.shape}')

    ###########################################################
    # your own model, prediction, and phenotyping code goes below here
    # the prediction code below is just an example, you should replace it with your own classifier
    #
    # X : contains the windowed accelerometer data with shape (num_windows, SAMPLE_RATE*WINDOW_SEC, 3)
    # T : is a 1D vector shape (num_windows,) that contains the start timestamp of each window in X
    #
    # use the dataframes provided by the template to store your output:
    # -  df_pred for the activity prediction dataframe
    # -  summary for the summary statistics / phenotype dataframe
    ###########################################################

    # feature extraction
    X_feats = pd.DataFrame([extract_features(x) for x in X])

    # Load sklearn model
    # https://scikit-learn.org/stable/modules/model_persistence.html
    random_forest = joblib.load('your_random_forest.joblib')

    # do your prediction work here.
    # y_pred should be a 1D vector shape (num_windows,), containing the predicted labels
    y_pred = random_forest.predict(X_feats)

    # construct a dataframe with time as the index and the labels in the 'prediction' column
    df_pred = pd.DataFrame({
        'prediction': y_pred
    }, index=T)

    # Reindex for missing values
    # The non-wear detection can cause some data to be NaN (periods of non-wear).
    # These values were taken out of X (and T) by df_to_windows (because we can't classify NaN values).
    # df_pred will therefore contain gaps in the time index where the non-wear was.
    # Reindexing fills up these gaps with NaN values and create a uniform continuous time index from start to end.
    newindex = pd.date_range(data_start, data_end, freq='{s}S'.format(s=WINDOW_SEC))
    df_pred = df_pred.reindex(newindex, method='nearest', fill_value=np.nan, tolerance='5S')

    # do your phenotype / summary statistics work on the predicted time series here
    # (you could also do it before reindexing, it depends on how you will handle missing values in the predicted series)
    # add your phenotypes as a column to the 'summary' dataframe (keep 'eid' as the first column)
    # Your new phenotypes should be scalar values, add each one as a separate column ('summary' should only have 1 row)

    summary = pd.DataFrame({
        'eid': pid
    }, index=[0])

    summary['my_sleep_phenotype'] = 0.5  # just an example, do actual phenotyping on the predicted time series

    ###########################################################
    # don't edit below this line.
    # REMINDER: make sure your df_pred and summary dataframes are in the same format as given by the template!
    ###########################################################
    print('Done')
    # save dataframes to disk (for later inspection)
    print('Saving dataframe')

    if group:
        path = os.path.join(OUTPUT_PATH, group)
    else:
        path = OUTPUT_PATH

    Path(path).mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(os.path.join(path, pid + '.csv'), index_label='timestamp')
    info.to_csv(os.path.join(path, pid + '_info.csv'), index=False)
    summary.to_csv(os.path.join(path, pid + '_summary.csv'), index=False)

    end_time = datetime.now()
    print(f'Duration: {end_time - start_time}')
