import pandas as pd
import numpy as np
import torch


def simple_CNNinput_to_RNNinput(cnn_input, save=True):
    n_features = cnn_input.shape[1]
    rnn_input = cnn_input.reshape((3609, 100, n_features))
    np.save(dir + "rnn_input.npy", rnn_input)
    return rnn_input

class RNNDataset(Dataset):
    def __init__(self, file_path, labels_file_path):
        self.data = np.load(file_path)
        self.labels = np.load(labels_file_path)
        self.dtype = torch.float32

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Assuming you have labels for each sample
        label = self.labels[idx]  # You should adjust this to fetch labels if available
        label = torch.tensor([label], dtype=self.dtype)
        sample = torch.tensor(sample, dtype=self.dtype)
        # sample = torch.unsqueeze(sample, dim=0)

        return sample, label


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = nhidden  # number of hidden states
        self.n_layers = 3  # number of LSTM layers (stacked)
        self.dropout = torch.nn.Dropout(0.2)

        self.l_lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True,
        )
        self.l_linear1 = torch.nn.Linear(
            self.n_hidden * self.seq_len, self.n_hidden * self.seq_len
        )
        self.l_linear2 = torch.nn.Linear(self.n_hidden * self.seq_len, 1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden = (hidden_state, cell_state)
        lstm_out, self.hidden = self.l_lstm(x, hidden)

        x = lstm_out.contiguous().view(batch_size, -1)
        x = self.dropout(x)
        x = self.l_linear1(x)
        # self.activation(x)
        x = self.dropout(x)
        x = self.l_linear2(x)

        return x
