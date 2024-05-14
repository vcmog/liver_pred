import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import utils.config as config
import csv

# Set paths
output_dir = config.output_dir
data_dir = config.data_dir
experiment_dir = output_dir / "leadtime_experiment_CVresults"


model_names = {
    "cnn_cv_results": "CNN",
    "current_cv_results": "Current",
    "current+trend_cv_results": "Current + Trend",
    "rnn_cv_results": "RNN",
    "trend_cv_results": "Trend",
}

plt.rcParams.update({"font.size": 12})
# Load the data
lead_times = range(0, 52, 4)


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

palette = ["#104862", "#A799B7", "#550C18", "#B5BD89", "#E55812"]

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
color_Index = 0
for model_name in model_names:
    plt.errorbar(
        lead_times,
        auc_means[model_name],
        yerr=[auc_lower[model_name], auc_upper[model_name]],
        label=model_names[model_name],
        elinewidth=1,
        capsize=3,
        color=palette[color_Index],
    )
    color_Index += 1
plt.legend()
plt.xlabel("Lead Time (weeks)")
plt.ylabel("AUC")
plt.title("AUC vs Lead Time")
plt.ylim(0 - 0.01, 1 + 0.01)
plt.savefig(experiment_dir / "figures/auc_vs_lead_time.png")

# Plot precisions
plt.figure(figsize=(10, 5))
color_Index = 0
for model_name in model_names.keys():
    plt.errorbar(
        lead_times,
        precision_means[model_name],
        yerr=[precision_lower[model_name], precision_upper[model_name]],
        label=model_names[model_name],
        elinewidth=1,
        capsize=3,
        color=palette[color_Index],
    )
    color_Index += 1
plt.legend()
plt.xlabel("Lead Time (weeks)")
plt.ylabel("Precision")
plt.title("Precision vs Lead Time")
plt.ylim(0 - 0.01, 1 + 0.01)
plt.savefig(experiment_dir / "figures/precision_vs_lead_time.png")

# Plot recalls
plt.figure(figsize=(10, 5))
color_Index = 0
for model_name in model_names:
    plt.errorbar(
        lead_times,
        recall_means[model_name],
        yerr=[recall_lower[model_name], recall_upper[model_name]],
        label=model_names[model_name],
        elinewidth=1,
        capsize=3,
        color=palette[color_Index],
    )
    color_Index += 1
plt.legend()
plt.xlabel("Lead Time (weeks)")
plt.ylabel("Recall")
plt.title("Recall vs Lead Time")
plt.ylim(0 - 0.01, 1 + 0.01)
plt.savefig(experiment_dir / "figures/recall_vs_lead_time.png")

# Plot average precisions
plt.figure(figsize=(10, 5))
color_Index = 0
for model_name in model_names:
    plt.errorbar(
        lead_times,
        average_precision_means[model_name],
        yerr=[
            average_precision_lower[model_name],
            average_precision_upper[model_name],
        ],
        label=model_names[model_name],
        elinewidth=1,
        capsize=3,
        color=palette[color_Index],
    )
    color_Index += 1
plt.legend()
plt.xlabel("Lead Time (weeks)")
plt.ylabel("Average Precision")
plt.title("Average Precision vs Lead Time")
plt.ylim(0 - 0.01, 1 + 0.01)
plt.savefig(experiment_dir / "figures/average_precision_vs_lead_time.png")


# load results over time for the current model and find the sum of the confusion matrix
population_sizes = {}

for model_name in model_names:
    population_sizes[model_name] = []
    for lead_time in lead_times:
        lead_time_folder = experiment_dir / f"leadtime={lead_time}weeks"
        results = pd.read_csv(lead_time_folder / f"{model_name}.csv", index_col=0)
        results.drop(columns="index", inplace=True)
        conf_matrix = results["conf_matrix"]
        # print(conf_matrix)
        # print(conf_matrix.values[0].split("\n"))
        x = csv.reader(
            conf_matrix.values[0].replace("[", "").replace("]", "").split("\n")
        )
        # print(np.array(list(x)))
        # convert the strings to an np array of integers
        conf_matrix_0 = sum(
            [int(a) if a != "" else 0 for a in list(x)[0][0].lstrip(" ").split(" ")]
        )
        x = csv.reader(
            conf_matrix.values[0].replace("[", "").replace("]", "").split("\n")
        )
        conf_matrix_1 = sum(
            [int(a) if a != "" else 0 for a in list(x)[1][0].lstrip(" ").split(" ")]
        )
        population_sizes[model_name].append(conf_matrix_0 + conf_matrix_1)

# Plot population sizes
plt.figure(figsize=(10, 2), dpi=100)
color_Index = 0
linestyle = ["-", "--", ":"]
for model_name in model_names:
    if model_name == "rnn_cv_results":
        label = "RNN"
    elif model_name == "cnn_cv_results":
        label = "CNN"
    elif model_name == "current_cv_results":
        label = "Feature Eng"
    else:
        continue

    plt.plot(
        lead_times,
        population_sizes[model_name],
        label=model_names[model_name],
        color=palette[color_Index],
        linestyle=linestyle[color_Index],
    )
    color_Index += 1
plt.legend()
plt.xlabel("Lead Time (weeks)", fontsize=14)
plt.ylabel("Test Set Size", fontsize=14)
plt.title("Test Set Size vs Lead Time", fontsize=14)
plt.savefig(experiment_dir / "figures/test_set_size_vs_lead_time.png", dpi=300)


# FInd mean of each metric at leadtime = 0 and make into a dataframe with each metric as the column and the model name as the row
lead_time = 0
metrics_mean = {}
for model_name in model_names:
    results = pd.read_csv(
        experiment_dir / f"leadtime={lead_time}weeks" / f"{model_name}.csv", index_col=0
    )
    results.drop(columns=["index", "conf_matrix"], inplace=True)
    metrics_mean[model_name] = results.mean()

df_metrics_mean = pd.DataFrame(metrics_mean).T

# Plot the metrics at leadtime = 0
plt.figure(figsize=(7, 4))
color_Index = 0
metrics = ["precision", "recall", "auroc", "average_precision"]
metric_label = {
    "precision": "precision",
    "recall": "recall",
    "auroc": "auroc",
    "average_precision": "avg_precision",
}
bar_width = 0.2
index = np.arange(len(model_names))
for i, metric in enumerate(metrics):
    plt.bar(
        index + i * bar_width,
        df_metrics_mean[metric],
        bar_width,
        label=metric_label[metric],
        color=palette[color_Index],
    )
    color_Index += 1

plt.xticks(index + bar_width * (len(metrics) - 1) / 2, model_names.values())
plt.legend(bbox_to_anchor=(0.8, 1))
plt.xlabel("Model")
plt.ylabel("Metric")
plt.title("Metrics at Lead Time = 0")
plt.savefig(experiment_dir / "figures/metrics_at_lead_time_0.png")


# Print metrics for RNN at leadtime=24 and leadtime=52
results_24 = pd.read_csv(
    experiment_dir / "leadtime=24weeks" / f"rnn_cv_results.csv",
    index_col=0,
    dtype={
        "precision": float,
        "recall": float,
        "auroc": float,
        "average_precision": float,
        "conf_matrix": str,
    },
)
results_52 = pd.read_csv(
    experiment_dir / "leadtime=52weeks" / f"rnn_cv_results.csv",
    index_col=0,
    dtype={
        "precision": float,
        "recall": float,
        "auroc": float,
        "average_precision": float,
        "conf_matrix": str,
    },
)

print(
    [
        (column, results_24[column].mean()) if column != "conf_matrix" else 0
        for column in results_24.columns
    ]
)
print(
    [
        (column, results_52[column].mean()) if column != "conf_matrix" else 0
        for column in results_52.columns
    ]
)

plt.show()
