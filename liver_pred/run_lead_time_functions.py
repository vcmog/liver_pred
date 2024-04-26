import pandas as pd
import numpy as np
import utils.preprocessing.feature_generation as fg
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression
import utils.preprocessing.missing_data as md
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
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
from utils.Models import models, pytorch_models
from utils.Evaluation import evaluation
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from pathlib import Path

dir = r"C:\Users\victo\OneDrive - University of Leeds\Documents\Uni Work\Project\MIMIC Work\Liver Cancer Prediction\liver_pred\data\interim"
model_dir = r"C:\Users\victo\OneDrive - University of Leeds\Documents\Uni Work\Project\MIMIC Work\Liver Cancer Prediction\liver_pred\data\models"

# Experiment Settings
lead_time = 0
max_history = 365 * 2
nhidden = 3
experiment_dir = r"C:\Users\victo\OneDrive - University of Leeds\Documents\Uni Work\Project\MIMIC Work\Liver Cancer Prediction\liver_pred\outputs\leadtime={}".format(
    lead_time
)
output_dir = r"C:\Users\victo\OneDrive - University of Leeds\Documents\Uni Work\Project\MIMIC Work\Liver Cancer Prediction\liver_pred\outputs\leadtime_experiment_CVresults\leadtime={}".format(
    lead_time
)

# Load Data
cohort_ids = pd.read_csv(dir + r"\matched_cohort_ids.csv", index_col=0)
processed_labs = pd.read_csv(
    dir + r"\processed_lab_data.csv",
    parse_dates=["charttime", "index_date"],
    index_col=0,
)

# Load Models


def run_FEng_model(feature_dfs, lead_time=0, mode="current+trend"):
    """
    Runs the feature engineering model for liver cancer prediction.

    Args:
        processed_labs (pd.DataFrame): The processed lab data.
        mode (str, optional): The mode for feature engineering. Must be one of 'current+trend', 'trend', 'current'. Defaults to "current+trend".

    Raises:
        ValueError: If an invalid mode is provided.

    Returns:
        None
    """

    if mode == "current+trend":
        feature_df = feature_dfs[2]
    elif mode == "trend":
        feature_df = feature_dfs[1].merge(
            feature_dfs[0][["subject_id", "outcome"]], on="subject_id"
        )
    elif mode == "current":
        feature_df = feature_dfs[0]
    else:
        raise ValueError(
            "Invalid mode. Must be one of 'current+trend', 'trend', 'current'"
        )

    y = feature_df["outcome"]
    X = feature_df.drop(columns=["outcome", "subject_id"])

    # Perform cross-validation
    cv_results_train = []
    cv_results_test = []
    for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True).split(
        X, y
    ):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale features
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train))
        X_test = pd.DataFrame(scaler.transform(X_test))
        X_train = X_train.set_axis(X.columns, axis=1).fillna(0)
        X_test = X_test.set_axis(X.columns, axis=1).fillna(0)

        # Load model
        print("Loading model...")
        current_trend_model = joblib.load(
            experiment_dir + r"\{}_nnmodel.pkl".format(mode)
        )

        # Fit model
        print("Fitting model...")
        current_trend_model.fit(X_train, y_train)

        # Evaluate model
        print("Evaluating model...")
        train_preds = current_trend_model.predict_proba(X_train)[:, 1]
        train_results = evaluation.evaluate_performance_nontorch(train_preds, y_train)
        train_results["index"] = lead_time
        cv_results_train.append(train_results)

        test_preds = current_trend_model.predict_proba(X_test)[:, 1]
        test_results = evaluation.evaluate_performance_nontorch(test_preds, y_test)
        test_results["index"] = lead_time
        cv_results_test.append(test_results)

    # Save cross-validation results
    cv_results_df = pd.DataFrame(cv_results_test)

    cv_results_df.to_csv(output_dir + r"\{}_cv_results.csv".format(mode), index=False)

    print("'{}' Model Complete.".format(mode))


def run_RNN_model(processed_labs, lead_time=0, max_history=max_history, k_folds=5):

    rnn_input, y = fg.create_array_for_RNN(
        processed_labs, lead_time=lead_time, max_history=max_history
    )

    np.save(dir + "rnn_input.npy", rnn_input)
    np.save(dir + "rnn_output.npy", y.flatten())
    print("files saved successfully.")

    dataset = pytorch_models.RNNDataset(dir + "rnn_input.npy", dir + "rnn_output.npy")
    n_features = rnn_input.shape[2]
    seq_length = rnn_input.shape[1]

    class_weights = torch.tensor(
        [len(y) / (2 * sum(y == 1)), len(y) / (2 * sum(y == 0))]
    )

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True)

    cv_results_train = []
    cv_results_test = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        # split train into 90,10 for validation
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, shuffle=True)

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # Compute the mean and standard deviation of the dataset
        mean_sum = 0.0
        std_sum = 0.0
        total_samples = 0
        for data, _ in train_loader:
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
        train_loader.transform = custom_transform
        test_loader.transform = custom_transform
        val_loader.transform = custom_transform

        # Load model
        model = pytorch_models.RNN(
            n_features=n_features, seq_length=seq_length, nhidden=nhidden
        )

        # Train model
        pytorch_models.train_model(
            model,
            train_loader,
            val_loader,
            100,
            lr=0.01,
            pos_class_weight=class_weights[0],
            save_dir=None,
        )

        train_results = evaluation.evaluate_performance_torchmodel(model, train_loader)
        train_results["index"] = lead_time
        # Evaluate model
        test_results = evaluation.evaluate_performance_torchmodel(model, test_loader)
        test_results["index"] = lead_time

        # Save results
        cv_results_train.append(train_results)
        cv_results_test.append(test_results)

    # Save cross-validation results
    cv_results_df = pd.DataFrame(cv_results_test)
    cv_results_df.to_csv(output_dir + r"\rnn_cv_results.csv", index=False)

    print("RNN Model Complete.")


def run_CNN_model(processed_labs, lead_time=0, max_history=max_history, k_folds=5):
    cnn_input, y = fg.create_array_for_CNN(
        processed_labs, lead_time=lead_time, max_history=max_history
    )

    np.save(dir + "cnn_input.npy", cnn_input)
    np.save(dir + "cnn_output.npy", y.flatten())
    print("files saved successfully.")

    dataset = pytorch_models.OneD_Dataset(dir + "cnn_input.npy", dir + "cnn_output.npy")

    class_weights = torch.tensor(
        [len(y) / (2 * sum(y == 1)), len(y) / (2 * sum(y == 0))]
    )

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True)

    cv_results_train = []
    cv_results_test = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        # split train into 90,10 for validation
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, shuffle=True)

        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # Compute the mean and standard deviation of the dataset
        mean_sum = 0.0
        std_sum = 0.0
        total_samples = 0
        for data, _ in train_loader:
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
        train_loader.transform = custom_transform
        test_loader.transform = custom_transform
        val_loader.transform = custom_transform

        # Load model
        model = pytorch_models.onedCNN()

        # Train model
        pytorch_models.train_model(
            model,
            train_loader,
            val_loader,
            100,
            lr=0.0001,
            pos_class_weight=class_weights[0],
            save_dir=None,
        )

        train_results = evaluation.evaluate_performance_torchmodel(model, train_loader)
        train_results["index"] = lead_time
        # Evaluate model
        test_results = evaluation.evaluate_performance_torchmodel(model, test_loader)
        test_results["index"] = lead_time

        # Save results
        cv_results_train.append(train_results)
        cv_results_test.append(test_results)

    # Save cross-validation results
    cv_results_df = pd.DataFrame(cv_results_test)
    cv_results_df.to_csv(output_dir + r"\cnn_cv_results.csv", index=False)

    print("CNN Model Complete.")


# Get feature dfs
feature_dfs = fg.generate_features(
    processed_labs, lead_time=lead_time, max_history=max_history
)


# Run Models
run_FEng_model(feature_dfs, mode="current+trend")
run_FEng_model(feature_dfs, mode="trend")
run_FEng_model(feature_dfs, mode="current")

run_RNN_model(processed_labs, lead_time=lead_time, max_history=max_history, k_folds=5)

run_CNN_model(processed_labs, lead_time=lead_time, max_history=max_history, k_folds=5)
