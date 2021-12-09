''' Template for submission: Example for RF + LSTM
You would need to provide this code + `your_random_forest_model.joblib` +
`your_lstm_model.pth`
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from tqdm.auto import tqdm

import utils


def predict(datafile):
    '''
    Input
    -----
    `datafile`: A csv file with columns `time, x, y, z`

    Return
    ------
    `y`: numpy array of predicted labels

    '''

    # Load data
    X, T = make_windows(load_data(datafile))

    # Extract features
    X_feats = pd.DataFrame([extract_features(x) for x in X])

    # Load sklearn model
    # https://scikit-learn.org/stable/modules/model_persistence.html
    random_forest = joblib.load('your_random_forest_model.joblib')

    # Load PyTorch model
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html)
    lstm = LSTM()
    lstm.load_state_dict(torch.load('your_lstm_model.pth'))

    # Classify
    Y = random_forest.predict_proba(X_feats).astype('float32')
    Y = F.softmax(Y, dim=1)  # convert to probabilities
    y = torch.argmax(Y, dim=1)  # convert to classes
    y = y.numpy()  # cast to numpy array

    return y


def load_data(datafile):
    """ Utility function to load the data files with correct dtypes """
    data = pd.read_csv(
        datafile,
        usecols=['time', 'x', 'y', 'z'],
        index_col='time', parse_dates=['time'],
        dtype={'x': 'f4', 'y': 'f4', 'z': 'f4'}
    )
    return data


def make_windows(data, winsec=30, sample_rate=100, verbose=False):

    X, T = [], []

    for t, w in tqdm(data.resample(f"{winsec}s", origin='start'), disable=not verbose):

        if len(w) < 1:
            continue

        t = t.to_numpy()

        x = w[['x', 'y', 'z']].to_numpy()

        if not is_good_window(x, sample_rate, winsec):  # skip if bad window
            continue

        X.append(x)
        T.append(t)

    X = np.stack(X)
    T = np.stack(T)

    return X, T


def is_good_window(x, sample_rate, winsec):
    ''' Check there are no NaNs and len is good '''

    # Check window length is correct
    window_len = sample_rate * winsec
    if len(x) != window_len:
        return False

    # Check no nans
    if np.isnan(x).any():
        return False

    return True


def extract_features(xyz):
    ''' Extract features. xyz is an array of shape (N,3) '''

    feats = {}
    feats['xMean'], feats['yMean'], feats['zMean'] = np.mean(xyz, axis=0)
    feats['xStd'], feats['yStd'], feats['zStd'] = np.std(xyz, axis=0)
    v = np.linalg.norm(xyz, axis=1)  # magnitude stream
    feats['mean'], feats['std'] = np.mean(v), np.std(v)

    return feats


class LSTM(nn.Module):
    ''' Single-layer bidirectional LSTM '''
    def __init__(self, input_size=5, output_size=5, hidden_size=1024):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.hidden2output = nn.Linear(2*hidden_size, output_size)

    def forward(self, sequence):
        hiddens, (hidden_last, cell_last) = self.lstm(
            sequence.view(len(sequence), -1, self.input_size))
        output = self.hidden2output(
            hiddens.view(-1, hiddens.shape[-1])).view(
                hiddens.shape[0], hiddens.shape[1], self.output_size
        )
        return output


if __name__ == "__main__":
    ''' Example '''
    y = predict('P123.csv')
    np.save('my_predictions.npy', y)
