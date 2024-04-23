import pandas as pd
import numpy as np
import torch

torch.set_default_dtype(torch.float32)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
from utils.config import data_dir
from torchvision import transforms

import matplotlib.pyplot as plt


class OneD_Dataset(Dataset):
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

        return sample, label


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


class onedCNN(nn.Module):
    def __init__(self):
        super(onedCNN, self).__init__()
        # Define your convolutional layers
        self.conv1 = nn.Conv1d(in_channels=45, out_channels=90, kernel_size=3)
        # (input_size - kernel_size + 2*padding)/stride + 1
        # for 54 features: 108
        self.conv2 = nn.Conv1d(
            in_channels=90, out_channels=180, kernel_size=3, padding=1
        )
        # for 54 features: 216
        self.conv3 = nn.Conv1d(
            in_channels=180, out_channels=180, kernel_size=3, padding=1
        )
        # for 54 features: 216
        self.conv4 = nn.Conv1d(
            in_channels=180, out_channels=360, kernel_size=2, padding=1
        )
        # Define your fully connected layers

        self.fc1 = nn.Linear(360 * 7, 64)
        self.fc2 = nn.Linear(64, 1)  # Assuming you have 2 classes

        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        # Input x has shape (batch_size, channels, height, width)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=2)

        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=1, stride=2)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # x.size(0) is the batch size

        x = self.dropout(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length, nhidden):
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


def initialise_dataloaders(dataset, batch_size, shuffle=True, num_workers=0):

    # Split the dataset indices into training and testing sets
    train_size = int(0.7 * len(dataset))  # 80% for training, adjust ratio as needed
    test_size = int(0.2 * len(dataset))  # 20% for testing, adjust ratio as needed
    val_size = len(dataset) - (
        train_size + test_size
    )  # 10% of the training data for validation
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create separate DataLoaders for training and testing
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    # Compute the mean and standard deviation of the dataset
    mean_sum = 0.0
    std_sum = 0.0
    total_samples = 0

    for data, _ in train_dataloader:
        total_samples += data.size(0)
        mean_sum += data.mean(
            dim=(0, 1)
        )  # Calculate mean along batch (0), width (2), and height (3) axes
        std_sum += torch.std(
            data, dim=(0, 1)
        )  # Calculate std along batch (0), width (2), and height (3) axes
    mean = mean_sum / total_samples
    std = std_sum / total_samples

    custom_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert data to PyTorch tensor
            # nan,  # Handle NaN values
            transforms.Normalize(
                mean=[mean], std=[std]
            ),  # Normalize data using computed mean and std
        ]
    )
    train_dataloader.transform = custom_transform
    test_dataloader.transform = custom_transform
    val_dataloader.transform = custom_transform
    return train_dataloader, test_dataloader, val_dataloader


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs=10,
    lr=0.001,
    pos_class_weight=None,
):
    # Initialize your CNN

    best_val_loss = float("inf")

    # Define your loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_class_weight, reduction="mean")
    unweighted_criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    training_losses = []
    val_losses = []

    # Train the model
    for epoch in range(num_epochs):
        # Create a progress bar
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        running_loss = 0.0

        # Iterate over the dataset
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            # Update the progress bar with the current loss
            progress_bar.set_postfix({"loss": running_loss / len(progress_bar)})

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_dataloader)
        training_losses.append(epoch_loss)
        # Validation phase
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = unweighted_criterion(outputs, labels)
                running_val_loss += loss.item()

        # Calculate average validation loss for the epoch
        epoch_val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), data_dir / "1d_model.pth")

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}",
            "Validation Loss:",
            epoch_val_loss,
        )

        plt.plot(training_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
