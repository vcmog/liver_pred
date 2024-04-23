import pandas as pd
import torch
from sklearn.metrics import (
    precision_score,
    average_precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt


# Function to compute evaluation metrics
def evaluate_performance_torchmodel(model, dataloader, plot_results=False):
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
    average_precision = average_precision_score(true_labels, all_probabilities)
    if plot_results:
        # plot ROC curve
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        # plt.show()

    results_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "conf_matrix": conf_matrix,
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "average_precision": average_precision,
    }
    return accuracy, precision, recall, conf_matrix, auroc


def evaluate_performance_nontorch(y_probs, y_true, print=False, threshold=0.5):
    """
    Evaluate the performance of a binary classification model by calculating various metrics.

    Parameters:
    - y_probs (array-like): Predicted probabilities for the positive class.
    - y_true (array-like): True labels for the samples.

    Returns:
    - f1 (float): F1 score.
    - roc_auc (float): ROC AUC score.
    - precision (float): Precision score.
    - recall (float): Recall score.
    """
    y_pred = y_probs > threshold

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, y_probs)
    tpr, fpr, thresholds = roc_curve(y_true, y_probs)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    average_precision = average_precision_score(y_true, y_probs)

    confusion_matrix = confusion_matrix(y_true, y_pred)

    if print:
        print(f"F1 Score: {f1}")
        print(f"ROC AUC: {roc_auc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("Confusion Matrix:", confusion_matrix)

    results_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "conf_matrix": confusion_matrix,
        "auroc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "average_precision": average_precision,
    }
    return f1, roc_auc, precision, recall
