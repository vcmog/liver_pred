# general
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# my packaes
import utils.preprocessing.feature_generation as fg
import utils.preprocessing.missing_data as md

# data preparation
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# evaluation
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# confusion_matrix, ConfusionMatrixDisplay,roc_auc_score, average_precision_score, f1_score, classification_report, precision_score, recall_score, roc_curve
from sklearn.model_selection import cross_validate
from sklearn.utils import resample

# utils
import joblib
from sys import getsizeof

# from scipy.stats import linregress


def prepare_features(
    feature_df,
    fill_na=True,
    sparse_col_threshold=None,
    scale=True,
    test_size=0.2,
    random_state=42,
):
    """
    Prepare features for training and testing a machine learning model. Scale before imputing missing values.

    Args:
        feature_df (DataFrame): The input DataFrame containing the features and outcome variable.
        fill_na (bool, optional): Whether to fill missing values with 0. Defaults to True.
        sparse_col_threshold (float, optional): The threshold for removing sparse columns. Defaults to None.
        scale (bool, optional): Whether to scale the features. Defaults to True.
        test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.2.
        random_state (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing the scaled training features, scaled testing features, training outcome variable, and testing outcome variable.
    """

    print("Preparing Features")

    y = feature_df["outcome"]
    X = feature_df.drop(columns=["outcome"]).set_index("subject_id")

    if sparse_col_threshold:
        sparse_cols = X.columns[X.isnull().mean() > sparse_col_threshold]
        X = X.drop(columns=sparse_cols)

    train_size = 1 - test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    print(
        f"Train Length: {len(X_train)}        Train cases: {len(y_train[y_train==1])}    Proportion: {len(y_train[y_train==1])/len(y_train)*100} %"
    )
    print(
        f"Test Length: {len(X_test)}          Test cases: {len(y_test[y_test==1])}       Proportion: {len(y_test[y_test==1])/len(y_test)*100} %"
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
    X_train_scaled = X_train_scaled.set_axis(X_train.columns, axis=1)

    if fill_na:
        X_train_scaled = X_train_scaled.fillna(0, inplace=False)

    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    X_test_scaled = X_test_scaled.fillna(0, inplace=False)
    X_test_scaled = X_test_scaled.set_axis(X_train.columns, axis=1)
    return X_train_scaled, X_test_scaled, y_train, y_test


models = {
    "LR": LogisticRegression(max_iter=1000, solver="saga"),
    "RF": RandomForestClassifier(n_jobs=-1),  # class_weight=class_weights,
    "GB": GradientBoostingClassifier(),
    "NN": MLPClassifier(),
    "SVM": SVC(probability=True),
}


def tune_hyperparameters_grid(
    model, param_grid, X_train, y_train, save_result=False, model_dir=None
):
    """
    Tune hyperparameters of a model using grid search.

    Args:
        model: The model to be tuned.
        param_grid: The parameter grid to search over.
        X_train: The training data.
        y_train: The target labels.
        save_result: Whether to save the best parameters and model.
        model_dir: The directory to save the best parameters and model.

    Returns:
        y_pred: The predicted labels on the training data.
        y_prob: The predicted probabilities on the training data.
        clf: The fitted grid search classifier.
    """
    print(f"Running {model}")
    clf = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring="average_precision")
    clf.fit(X_train, y_train)
    if save_result:
        joblib.dump(clf, model_dir / r"\bestparams_" + f"{model}.pkl")
    y_pred = clf.predict(X_train)
    y_prob = clf.predict_proba(X_train)[:, 1]
    return y_pred, y_prob, clf


def tune_hyperparameters_random(
    model, dists, X_train, y_train, save_result=False, model_dir=None
):
    """
    Tune hyperparameters of a machine learning model using random search.

    Parameters:
    - model: The machine learning model to tune.
    - dists: A dictionary of parameter distributions for random search.
    - X_train: The training data.
    - y_train: The target labels.
    - save_result: Whether to save the best parameters to a file (default: False).
    - model_dir: The directory to save the best parameters file (default: None).

    Returns:
    - The best parameters found during the random search.

    """
    print(f"Running {model}")
    clf = RandomizedSearchCV(model, dists, random_state=0)
    clf.fit(X_train, y_train)
    if save_result:
        joblib.dump(clf, model_dir / r"\bestparams_" + f"{model}.pkl")
    return clf.best_params_
