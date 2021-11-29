# %% [markdown]
'''
# Training on the spectrogram of the signal

Computing the [spectrogram](https://en.wikipedia.org/wiki/Spectrogram) of a
signal is a common visualization method in signal processing to treat
signals as 2D images. Recent works have looked at using the spectrogram
for classification, thus converting the signal recognition task into an image
recognition one and leveraging techniques from computer vision.

## Setup
'''

# %%
import os
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

import utils

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = True

# Grab a GPU if there is one
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using {} device: {}".format(device, torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using {}".format(device))

# %% [markdown]
'''
## Load dataset
'''

# %%

# Path to your extracted windows
DATASET_PATH = 'processed_data/'
print(f'Content of {DATASET_PATH}')
print(os.listdir(DATASET_PATH))

X = np.load(DATASET_PATH+'X.npy', mmap_mode='r')
Y = np.load(DATASET_PATH+'Y.npy')
T = np.load(DATASET_PATH+'T.npy')
pid = np.load(DATASET_PATH+'pid.npy')

# As before, let's map the text annotations to simplified labels
ANNO_LABEL_DICT_PATH = 'capture24/annotation-label-dictionary.csv'
anno_label_dict = pd.read_csv(ANNO_LABEL_DICT_PATH, index_col='annotation', dtype='string')
Y = anno_label_dict.loc[Y, 'label:Willetts2018'].to_numpy()

# Transform to numeric
le = LabelEncoder().fit(Y)
Y = le.transform(Y)

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
    X[mask_train], Y[mask_train], T[mask_train], pid[mask_train]
X_test, Y_test, T_test, pid_test = \
    X[mask_test], Y[mask_test], T[mask_test], pid[mask_test]
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)


# %% [markdown]
''' ## Visualization

Let's visualize the spectrograms of the acceleration norm for each activity class.
We use
[scipy.signal.stft](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html).
Here, the arguments `nperseg` and `noverlap` determine the size of the
resulting spectrogram. For `nperseg=120` and `noverlap=72`, the spectrogram
size is $61\times 61$.

'''

# %%
# Spectrogram parameters
N_FFT = 120
HOP_LENGTH = 48
WINDOW = 'hann'
NUM_PLOTS = 10

labels = np.unique(Y_train)
num_labels = len(labels)

fig, axs = plt.subplots(num_labels, NUM_PLOTS, figsize=(10,5))
for i in range(num_labels):
    axs[i,0].set_ylabel(le.inverse_transform([labels[i]]).item())
    idxs = np.where(Y_train==i)[0]
    for j in range(NUM_PLOTS):
        _, _, z = scipy.signal.stft(
            np.linalg.norm(X_train[idxs[j]], axis=1),  # acceleration vector norm
            nperseg=N_FFT,
            noverlap=N_FFT-HOP_LENGTH,
            window=WINDOW,
            boundary=None, padded=False
        )
        z = np.log(np.abs(z) + 1e-16)
        axs[i,j].imshow(z, cmap='coolwarm')
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])
fig.show()

# %% [markdown]
'''
## Architecture design

As a baseline, let's use a convolutional neural network (CNN) with a typical
pyramid-like structure. The input to the network is a `(N,3,61,61)` array,
corresponding to `N` spectrograms for each axis signal. Again, note the
*channels-first* format: `(3,61,61)` instead of `(61,61,3)`.

The output of the CNN is a `(N,num_labels)` array where each row contains
predicted unnormalized class scores or *logits*; pass each row to a softmax
if you want to convert it to probabilities.

'''

# %%
class ConvBNReLU(nn.Module):
    ''' Convolution + batch normalization + ReLU is a common trio '''
    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1, bias=True
    ):
        super(ConvBNReLU, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.main(x)

class CNN(nn.Module):
    ''' Typical CNN design with pyramid-like structure '''
    def __init__(self, output_size=5, in_channels=3, num_filters_init=8):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            ConvBNReLU(in_channels, num_filters_init,
            3, 2, 1, bias=False),  # 61x61 -> 31x31
            ConvBNReLU(num_filters_init, num_filters_init*2,
            3, 2, 1, bias=False),  # 31x31 -> 16x16
            ConvBNReLU(num_filters_init*2, num_filters_init*4,
            4, 2, 1, bias=False),  # 16x16 -> 8x8
            ConvBNReLU(num_filters_init*4, num_filters_init*8,
            4, 2, 1, bias=False),  # 8x8 -> 4x4
            ConvBNReLU(num_filters_init*8, num_filters_init*16,
            4, 1, 0, bias=False),  # 4x4 -> 1x1
            nn.Conv2d(num_filters_init*16, output_size,
            1, 1, 0, bias=True)
        )

    def forward(self, x):
        return self.cnn(x).view(x.shape[0],-1)

# %% [markdown]
'''
## Helper functions

'''

# %%

def create_dataloader(X, y=None, batch_size=1, shuffle=False):
    ''' Create a (batch) iterator over the dataset.
    Alternatively, use PyTorch's Dataset and DataLoader classes.
    See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html '''

    # Spectrogram parameters
    N_FFT = 120
    HOP_LENGTH = 48
    WINDOW = torch.hann_window(N_FFT)

    if shuffle:
        idxs = np.random.permutation(np.arange(len(X)))
    else:
        idxs = np.arange(len(X))
    for i in range(0, len(idxs), batch_size):
        idxs_batch = idxs[i:i+batch_size]
        X_batch = X[idxs_batch].astype('f4')  # PyTorch defaults to float32
        X_batch = np.transpose(X_batch, (0,2,1))  # channels first: (N,M,3) -> (N,3,M)
        X_batch = torch.from_numpy(X_batch)
        Z_batch = torch.stft(
            # Pack channel and batch dimensions.
            # Also upload the batch to GPU to compute the STFT.
            # Let the GPU do the work -- Gordon Ramsey
            X_batch.reshape(-1, X_batch.shape[-1]).to(device),
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=WINDOW.to(device),
            center=False,
            return_complex=False
        )
        # Unpack channel and batch dimensions
        Z_batch = Z_batch.view(*X_batch.shape[:2], *Z_batch.shape[1:])
        Z_batch = torch.log(torch.norm(Z_batch, dim=-1) + 1e-16)
        if y is None:
            yield Z_batch
        else:
            y_batch = torch.from_numpy(y[idxs_batch])
            y_batch = y_batch.to(device)  # upload to GPU for consistency
            yield Z_batch, y_batch


def forward_by_batches(cnn, X):
    ''' Forward pass model on a dataset.
    Do this by batches so that we don't blow up the memory. '''
    Y = []
    cnn.eval()
    with torch.no_grad():
        for z in create_dataloader(X, batch_size=1024, shuffle=False):  # do not shuffle here!
            z = z.to(device)
            Y.append(cnn(z))
    cnn.train()
    Y = torch.cat(Y)
    return Y


def evaluate_model(cnn, X, Y):
    Y_pred = forward_by_batches(cnn, X)  # scores
    loss = F.cross_entropy(Y_pred, torch.from_numpy(Y).to(device)).item()

    Y_pred = F.softmax(Y_pred, dim=1)  # convert to probabilities
    Y_pred = torch.argmax(Y_pred, dim=1)  # convert to classes
    Y_pred = Y_pred.cpu().numpy()  # cast to numpy array
    kappa = metrics.cohen_kappa_score(Y, Y_pred)

    return {'loss':loss, 'kappa':kappa, 'Y_pred':Y_pred}

# %% [markdown]
'''
## Hyperparameters, model instantiation, loss function and optimizer

'''

# %%
num_filters_init = 32  # initial num of filters -- see class definition
in_channels = 3  # num channels of the signal -- equal to 3 for our raw triaxial timeseries
output_size = num_labels  # number of classes (sleep, sedentary, etc...)
num_epoch = 5  # num of epochs (full loops though the training set)
lr = 3e-4  # learning rate
batch_size = 32  # size of the mini-batch

cnn = CNN(
    output_size=output_size,
    in_channels=in_channels,
    num_filters_init=num_filters_init
).to(device)
print(cnn)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=lr)

# %% [markdown]
'''
## Training

'''

# %%
kappa_history_test = []
loss_history_test = []
loss_history_train = []
for i in tqdm(range(num_epoch)):
    dataloader = create_dataloader(X_train, Y_train, batch_size, shuffle=True)
    losses = []
    for z, target in dataloader:
        z, target = z.to(device), target.to(device)
        cnn.zero_grad()
        output = cnn(z)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Logging -- track train loss
        losses.append(loss.item())

    # --------------------------------------------------------
    #       Evaluate performance at the end of each epoch
    # --------------------------------------------------------

    # Logging -- average train loss in this epoch
    loss_history_train.append(utils.ewm(losses))

    # Logging -- evalutate performance on test set
    results = evaluate_model(cnn, X_test, Y_test)
    loss_history_test.append(results['loss'])
    kappa_history_test.append(results['kappa'])

# %% [markdown]
''' ## Model performane '''

# %%
# Loss history
plt.close('all')
fig, ax = plt.subplots()
ax.plot(loss_history_train, color='C0', label='train loss')
ax.plot(loss_history_test, color='C1', label='test loss')
ax.set_ylabel('loss (CE)')
ax.set_xlabel('epoch')
ax = ax.twinx()
ax.plot(kappa_history_test, color='C2', label='kappa')
ax.set_ylabel('kappa')
ax.grid(True)
fig.legend()
fig.show()

# Report
Y_test_pred_lab = le.inverse_transform(results['Y_pred'])  # back to text labels
Y_test_lab = le.inverse_transform(Y_test)  # back to text labels
print('\nClassifier performance')
print('Out of sample:\n', metrics.classification_report(Y_test_lab, Y_test_pred_lab))
