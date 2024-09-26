"""
Perform inference with pretrained model on a UKB accelerometer file.

Arguments:
    Input accelerometer file. If CSV, it must have columns 'time', 'x', 'y', 'z'.

Example usage:
    python template.py sample.csv.gz

Output:
    Prediction DataFrame in {eid}_prediction.csv format, stored in outputs/ folder.
    A {eid}_summary.json file with the calculated summary phenotypes and statistics.
"""

import actipy
import argparse
import os
import json
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

    # add more of your own features here
    # feats['...'] = ...

    return feats


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isnull(obj):  # handles pandas NAType
            return np.nan
        return json.JSONEncoder.default(self, obj)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='My App', usage='Apply the model on an accelerometer file.')
    parser.add_argument('input_file', type=str, help='input file path')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs/', help='output directory path')
    args = parser.parse_args()

    before = datetime.now()

    # set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Input file: {args.input_file}")
    input_file = Path(args.input_file)

    # just file name without path and extensions
    bname = input_file.name.split('.')[0]

    # load the file (it must have headers 'time', 'x', 'y', 'z')
    data = pd.read_csv(
        input_file,
        parse_dates=['time'],
        index_col='time',
        dtype={'x': 'f4', 'y': 'f4', 'z': 'f4'}
    )

    # print first 5 lines of the file
    print(data.head(5))

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

    # # Load sklearn model
    # # https://scikit-learn.org/stable/modules/model_persistence.html
    # random_forest = joblib.load('your_random_forest.joblib')

    # # do your prediction work here.
    # # y_pred should be a 1D vector shape (num_windows,), containing the predicted labels
    # y_pred = random_forest.predict(X_feats)

    # this is just an example: random binary predictions
    y_pred = np.random.randint(2, size=len(X_feats))

    # construct a dataframe with time as the index and the labels in the 'prediction' column
    df_pred = pd.DataFrame({
        'prediction': y_pred
    }, index=T)

    # Reindex for missing values
    # The non-wear detection can cause some data to be NaN (periods of non-wear).
    # These values were taken out of X (and T) by df_to_windows (because we can't classify NaN values).
    # df_pred will therefore contain gaps in the time index where the non-wear was.
    # Reindexing fills up these gaps with NaN values and create a uniform continuous time index from start to end.
    newindex = pd.date_range(data_start, data_end, freq='{s}s'.format(s=WINDOW_SEC))
    df_pred = df_pred.reindex(newindex, method='nearest', fill_value=np.nan, tolerance='5s')

    # do your phenotype / summary statistics work on the predicted time series
    # here (you could also do it before reindexing, it depends on how you will
    # handle missing values in the predicted series) add your phenotypes as a
    # column to the 'summary' dataframe 
    summary = {
        'Filename': str(input_file),
        'StartTime': data_start.strftime("%Y-%m-%d %H:%M:%S"),
        'EndTime': data_end.strftime("%Y-%m-%d %H:%M:%S"),
        'TotalTime(days)': (data_end - data_start).total_seconds() / 86400,
    }

    # pretty print summary
    print("\nSummary:")
    for k, v in summary.items():
        print(f'{k}: {v}')

    # save results to disk
    print('\nSaving to disk...')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(output_dir / f'{bname}_prediction.csv', index_label='time')
    with open(output_dir / f'{bname}_summary.json', 'w') as f:
        json.dump(summary, f, indent=4, cls=NpEncoder)

    after = datetime.now()
    print('Done!')
    print(f'Duration: {after - before}')
