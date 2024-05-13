import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import utils.config as config

# Set paths
output_dir = config.output_dir
data_dir = config.data_dir
experiment_dir = output_dir / "leadtime_experiment_CVresults"


model_names = [
    "cnn_cv_results",
    "current_cv_results",
    "current+trend_cv_results",
    "rnn_cv_results",
    "trend_cv_results",
]

# Load the data
lead_times = range(0, 52, 4)
# print(lead_times)
# lead_time_means = {}
# lead_time_errors = {}
# for lead_time in lead_times:
#    results_dict = {}
#    lead_time_folder = experiment_dir / f"leadtime={lead_time}weeks"
#    for name in model_names:
#        model_name = name
#        results = pd.read_csv(lead_time_folder / f"{name}.csv", index_col=0)
#        results.drop(columns="index", inplace=True)
#        results_dict[model_name] = results

# mave df with mean, std_dev of the cross_val results
#    results_mean = pd.DataFrame()
#    results_std = pd.DataFrame()
#    for model_name, results in results_dict.items():
#        del results["conf_matrix"]
#        results_mean[model_name] = results.mean()
#        results_std[model_name] = results.std()

#    lead_time_means[lead_time] = results_mean
#    lead_time_errors[lead_time] = results_std

#    print(lead_time_means.values())


precision_means = {}
precision_stds = {}
precision_upper = {}
precision_lower = {}
recall_means = {}
recall_stds = {}
recall_upper = {}
recall_lower = {}
auc_means = {}
auc_stds = {}
auc_upper = {}
auc_lower = {}
average_precision_means = {}
average_precision_stds = {}
average_precision_upper = {}
average_precision_lower = {}


# Plot the results
for model_name in model_names:
    precision_means[model_name] = []
    precision_stds[model_name] = []
    precision_upper[model_name] = []
    precision_lower[model_name] = []
    recall_means[model_name] = []
    recall_stds[model_name] = []
    recall_upper[model_name] = []
    recall_lower[model_name] = []
    auc_means[model_name] = []
    auc_stds[model_name] = []
    auc_upper[model_name] = []
    auc_lower[model_name] = []
    average_precision_means[model_name] = []
    average_precision_stds[model_name] = []
    average_precision_upper[model_name] = []
    average_precision_lower[model_name] = []
    for lead_time in lead_times:
        lead_time_folder = experiment_dir / f"leadtime={lead_time}weeks"
        results = pd.read_csv(lead_time_folder / f"{model_name}.csv", index_col=0)
        results.drop(columns="index", inplace=True)
        precision_means[model_name].append(results["precision"].mean())
        precision_stds[model_name].append(results["precision"].std())
        precision_upper[model_name].append(
            results["precision"].max() - results["precision"].mean()
        )
        precision_lower[model_name].append(
            results["precision"].mean() - results["precision"].min()
        )
        recall_means[model_name].append(results["recall"].mean())
        recall_stds[model_name].append(results["recall"].std())
        recall_upper[model_name].append(
            results["recall"].max() - results["recall"].mean()
        )
        recall_lower[model_name].append(
            results["recall"].mean() - results["recall"].min()
        )
        auc_means[model_name].append(results["auroc"].mean())
        auc_stds[model_name].append(results["auroc"].std())
        auc_upper[model_name].append(results["auroc"].max() - results["auroc"].mean())
        auc_lower[model_name].append(results["auroc"].mean() - results["auroc"].min())
        average_precision_means[model_name].append(results["average_precision"].mean())
        average_precision_stds[model_name].append(results["average_precision"].std())
        average_precision_upper[model_name].append(
            results["average_precision"].max() - results["average_precision"].mean()
        )
        average_precision_lower[model_name].append(
            results["average_precision"].mean() - results["average_precision"].min()
        )

metrics = ["precision", "recall", "auc", "average_precision"]

# Plot AUCs
plt.figure(figsize=(10, 5))
for model_name in model_names:
    plt.errorbar(
        lead_times,
        auc_means[model_name],
        yerr=[auc_lower[model_name], auc_upper[model_name]],
        label=model_name,
        elinewidth=1,
        capsize=3,
    )
plt.legend()
plt.xlabel("Lead Time (weeks)")
plt.ylabel("AUC")
plt.title("AUC vs Lead Time")


# Plot precisions
plt.figure(figsize=(10, 5))
for model_name in model_names:
    plt.errorbar(
        lead_times,
        precision_means[model_name],
        yerr=[precision_lower[model_name], precision_upper[model_name]],
        label=model_name,
        elinewidth=1,
        capsize=3,
    )
plt.legend()
plt.xlabel("Lead Time (weeks)")
plt.ylabel("Precision")
plt.title("Precision vs Lead Time")


# Plot recalls
plt.figure(figsize=(10, 5))
for model_name in model_names:
    plt.errorbar(
        lead_times,
        recall_means[model_name],
        yerr=[recall_lower[model_name], recall_upper[model_name]],
        label=model_name,
        elinewidth=1,
        capsize=3,
    )
plt.legend()
plt.xlabel("Lead Time (weeks)")
plt.ylabel("Recall")
plt.title("Recall vs Lead Time")


# Plot average precisions
plt.figure(figsize=(10, 5))
for model_name in model_names:
    plt.errorbar(
        lead_times,
        average_precision_means[model_name],
        yerr=[
            average_precision_lower[model_name],
            average_precision_upper[model_name],
        ],
        label=model_name,
        elinewidth=1,
        capsize=3,
    )
plt.legend()
plt.xlabel("Lead Time (weeks)")
plt.ylabel("Average Precision")
plt.title("Average Precision vs Lead Time")
plt.show()
