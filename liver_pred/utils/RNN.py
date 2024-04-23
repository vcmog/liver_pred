import pandas as pd
import numpy as np
import torch

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
    roc_curve,
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
from utils.config import data_dir


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


def initialise_dataloaders(
    file_path, labels_file_path, batch_size, shuffle=True, num_workers=0
):

    # Create a dataset
    dataset_rnn = RNNDataset(file_path, labels_file_path)

    # Split the dataset indices into training and testing sets
    train_size = int(0.7 * len(dataset_rnn))  # 80% for training, adjust ratio as needed
    test_size = int(0.2 * len(dataset_rnn))  # 20% for testing, adjust ratio as needed
    val_size = len(dataset_rnn) - (
        train_size + test_size
    )  # 10% of the training data for validation
    train_dataset_rnn, val_dataset_rnn, test_dataset_rnn = (
        torch.utils.data.random_split(dataset_rnn, [train_size, val_size, test_size])
    )

    # Create separate DataLoaders for training and testing
    train_dataloader_rnn = DataLoader(
        train_dataset_rnn,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    test_dataloader_rnn = DataLoader(
        test_dataset_rnn,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    val_dataloader_rnn = DataLoader(
        val_dataset_rnn, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    # Compute the mean and standard deviation of the dataset
    mean_sum = 0.0
    std_sum = 0.0
    total_samples = 0

    for data, _ in train_dataloader_rnn:
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
    train_dataloader_rnn.transform = custom_transform
    test_dataloader_rnn.transform = custom_transform
    val_dataloader_rnn.transform = custom_transform
    return train_dataloader_rnn, test_dataloader_rnn, val_dataloader_rnn


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


# Function to compute evaluation metrics
def evaluate_performance(model, dataloader):
    """
    Evaluate the performance of a recurrent neural network model on a given dataset.

    Args:
        model (torch.nn.Module): The trained recurrent neural network model.
        dataloader (torch.utils.data.DataLoader): The data loader for the evaluation dataset.

    Returns:
        accuracy (float): The accuracy of the model on the evaluation dataset.
        precision (float): The precision score of the model on the evaluation dataset.
        recall (float): The recall score of the model on the evaluation dataset.
        conf_matrix (numpy.ndarray): The confusion matrix of the model on the evaluation dataset.
        auroc (float): The area under the receiver operating characteristic curve (AUROC) of the model on the evaluation dataset.
    """

    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0
    predicted_labels = []
    true_labels = []
    all_probabilities = []
    with torch.no_grad():  # No need to track gradients during evaluation
        for inputs, labels in dataloader:
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            predicted = (probabilities > 0.5).int()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            predicted_labels.extend(predicted.numpy())
            true_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.numpy())

    # Compute accuracy
    accuracy = total_correct / total_samples

    # Compute precision and recall
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # calculate false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(true_labels, all_probabilities)
    # Calculate AUROC
    auroc = roc_auc_score(true_labels, all_probabilities)

    # plot ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    # plt.show()

    return accuracy, precision, recall, conf_matrix, auroc
