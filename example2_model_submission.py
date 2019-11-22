''' Example code for submission: RF + LSTM
You would need to provide this code + `your_random_forest_model.joblib` +
`your_lstm_model.pth`
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from tqdm import tqdm
import utils


def predict(X):
    '''
    Input
    -----
    `X`: numpy array of shape `(N,3,3000)` where each row corresponds to 30
    seconds of raw tri-axial acceleration recorded at 100Hz.

    Return
    ------
    `y`: numpy array of shape `(N,)` with predicted values (0: sleep, 1:
    sedentary, 2: tasks-light, 3: walking, 4: moderate)
    '''

    # Extract hand-crafted features
    extractor = utils.Extractor()
    X_feats = []
    for i in tqdm(range(X.shape[0])):
        X_feats.append(extractor.extract(X[i]))
    X_feats = np.stack(X_feats)

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
    # First num_prev predictions are taken directly from the random forest classifier
    NUM_PREV = 2
    y = np.concatenate((random_forest.predict(X_feats[:NUM_PREV]), y))

    return y


class LSTM(nn.Module):
    ''' Simple LSTM '''
    def __init__(self, input_size=5, output_size=5, hidden_size=8):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

    def forward(self, sequence):
        _, (hidden_last, cell_last) = self.lstm(
            sequence.view(len(sequence), -1, self.input_size))
        output = self.hidden2output(hidden_last.view(-1, self.hidden_size))
        return output
