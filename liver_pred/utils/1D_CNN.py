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
