''' Template for submission: CNN + HMM
You would need to provide this code + `your_cnn_model.pth` +
`your_hmm_model.npz`
'''
import numpy as np
import torch
import torch.nn as nn
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
    ''' Example code to deploy the model '''
    X = np.load('X_raw_test.npy')
    y = predict(X)
    np.save('my_predictions_for_2020.npy', y)
