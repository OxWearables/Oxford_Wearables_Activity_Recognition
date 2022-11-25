''' Template for submission: Example for CNN + HMM
You would need to provide this code + `your_cnn_model.pth` +
`your_hmm_model.npz`
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

    # Load PyTorch model
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html)
    cnn = CNN()
    cnn.load_state_dict(torch.load('your_cnn_model.pth'))

    # Load HMM weights
    hmm = np.load('your_hmm_model.npz')
    prior, emission, transition = hmm['prior'], hmm['emission'], hmm['transition']

    # Classify
    cnn.eval()
    X = torch.from_numpy(X)
    with torch.no_grad():
        y = cnn(X).numpy()
    y = utils.viterbi(y, prior, emission, transition)

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


class CNN(nn.Module):
    ''' Actually this is just linear regression '''
    def __init__(self, in_channels=3, input_size=3000, output_size=5):
        super(CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_channels, output_size, input_size, 1, 1, bias=True),
        )

    def forward(self, x):
        return self.main(x).view(x.shape[0],-1)


if __name__ == "__main__":
    ''' Example '''
    y = predict('P123.csv')
    np.save('my_predictions.npy', y)
