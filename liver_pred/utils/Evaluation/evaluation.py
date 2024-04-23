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
def evaluate_performance_torchmodel(model, dataloader):
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
