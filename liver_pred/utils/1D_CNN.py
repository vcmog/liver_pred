import pandas as pd
import numpy as np
import utils.colonflag.feature_generation as fg
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import torch

torch.set_default_dtype(torch.float32)
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
from torchvision import transforms
from sklearn.metrics import (
    precision_score,
    average_precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)


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


def initialise_dataloaders(
    file_path, labels_file_path, batch_size, shuffle=True, num_workers=0
):

    # File path to the .npy file
    # file_path_1d = dir+'CNN_1d_input.npy'
    # labels_file_path_1d = dir+'CNN_1d_output.npy'
    # Create a dataset
    dataset = OneD_Dataset(file_path, labels_file_path)

    # Create a DataLoader to load the dataset
    batch_size = 100
    shuffle = True  # You can set it to True if you want to shuffle the data
    num_workers = 0  # Number of subprocesses to use for data loading (0 means the data will be loaded in the main process)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

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
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
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


class onedCNN2(nn.Module):
    def __init__(self):
        super(onedCNN2, self).__init__()
        # Define your convolutional layers
        self.conv1 = nn.Conv1d(in_channels=54, out_channels=108, kernel_size=3)
        # (input_size - kernel_size + 2*padding)/stride + 1
        # (100 - 3 + 2*1)/1 + 1 = 100
        # (54 - 1 +2*1)/1 + 1 = 56
        self.conv2 = nn.Conv1d(
            in_channels=108, out_channels=216, kernel_size=3, padding=1
        )
        # 50 - 3 + 2*1)/1 + 1 = 50
        self.conv3 = nn.Conv1d(
            in_channels=216, out_channels=216, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv1d(
            in_channels=216, out_channels=432, kernel_size=2, padding=1
        )
        # Define your fully connected layers

        self.fc1 = nn.Linear(432 * 7, 64)
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
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)

    training_losses = []
    val_losses = []
    # Assuming your input data is in the form of a PyTorch tensor
    # Here's how you can train your CNN
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
            torch.save(model.state_dict(), dir + "/1d_model.pth")

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}",
            "Validation Loss:",
            epoch_val_loss,
        )
