"""Plotting functions for evaluations."""

import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Inspecting individual predictions accross cells
# ------------------------------------------------------------------------------


def plot_predictions_vs_groundtruth(
    predictions: np.ndarray,
    groundtruth: np.ndarray,
    classnames: list,
    max_n_predictions: int = 20,
    output_dir: str = None,
):
    """(Bar)plot multiple predictions across cell types vs one ground truth.

    Args:
        predicitions (np.ndarray): Predictions for each cell type (n, C)
        groundtruth (np.ndarray): Groundtruth for each cell type (n, C)
        output_dir (str, optional): Directory to save plots. None if showing.
    """
    if predictions.shape[1] != groundtruth.shape[1]:
        raise ValueError("Predictions and groundtruth must have smae number of classes")
    if len(classnames) != predictions.shape[1]:
        raise ValueError("Expected same number of cell names as predictions")
    if predictions.shape[0] > max_n_predictions:
        raise ValueError(
            f"Too many predictions ({predictions.shape[0]}) to plot. Max is {max_n_predictions}"
        )
    n, C = predictions.shape

    fig_width = C * 2
    fig_height = (n + 1) * 4
    plt.figure(figsize=(fig_width, fig_height))

    # Iterate over each prediction and create a subplot
    for i in range(n):
        plt.subplot(n + 1, 1, i + 1)
        plt.bar(classnames, predictions[i, :], label="Prediction")
        plt.xlabel("Cell Type")
        plt.ylabel("ATAC prediction")
        plt.grid("on")

    # Show ground truth
    plt.subplot(n + 1, 1, n + 1)
    plt.bar(classnames, groundtruth[0, :], label="Groundtruth", color="darkgreen")
    plt.xlabel("Cell Type")
    plt.ylabel("Groundtruth")
    plt.grid("on")

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, "predictions_vs_groundtruth_barplot.png"))
    else:
        plt.show()


# ------------------------------------------------------------------------------
# Inspecting multiple predictions within cells
# ------------------------------------------------------------------------------


def plot_scatter_predictions_vs_groundtruths(
    predictions: np.ndarray,
    groundtruths: np.ndarray,
    classnames: list,
    output_dir: str = None,
):
    """Scatter plot of predictions vs groundtruths for multiple cell types.

    Args:
        predictions (np.ndarray): Predictions for each cell type (n, C).
        groundtruths (np.ndarray): Groundtruths for each cell type (n, C).
        classnames (list): Names of the cell types.
        output_dir (str, optional): Directory to save plots. None if showing.
    """
    if predictions.shape != groundtruths.shape:
        raise ValueError("Predictions and groundtruths must have the same shape")

    n, C = predictions.shape
    if C != len(classnames):
        raise ValueError(
            "Number of classnames must match the second dimension of predictions"
        )

    # Calculate the size of the grid
    grid_size = int(np.ceil(np.sqrt(C)))
    fig_width, fig_height = 6 * grid_size, 4 * grid_size
    plt.figure(figsize=(fig_width, fig_height))

    # Create a subplot for each class
    for i in range(C):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.scatter(groundtruths[:, i], predictions[:, i], color="tab:blue")
        plt.xlabel("Groundtruth")
        plt.ylabel("Prediction")
        plt.title(f"{classnames[i]}")
        plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, "predictions_vs_groundtruths_scatter.png"))
    else:
        plt.show()


# ------------------------------------------------------------------------------
# Inspecting Metrics
# ------------------------------------------------------------------------------


def plot_metrics(
    metrics_array: np.ndarray,
    metric_names: list,
    classnames: list,
    output_dir: str = None,
):
    """Plots a bar chart across cell types for the given metrics. Will create one
    subplot for each metric in the list.

    Args:
        metrics_array (np.ndarray): Array of metric values (M, C), where M is the
        number of metrics.
        metric_names (list): Names of the metrics (used for labeling).
        classnames (list): List of cell type names.
        output_dir (str, optional): Directory to save plots. None if showing.
    """
    M, C = metrics_array.shape
    if len(metric_names) != M:
        raise ValueError("Number of metric names must match the number of metrics")
    if C != len(classnames):
        raise ValueError("Length of classnames must match the number of cell types")

    # Adjusting figure size based on number of cell types and metrics
    fig_width = max(10, len(classnames) * 1.5)
    fig_height = 5 * M  # Height for each metric
    plt.figure(figsize=(fig_width, fig_height))

    # Creating subplots for each metric
    for i, metric_name in enumerate(metric_names):
        plt.subplot(M, 1, i + 1)
        plt.bar(classnames, metrics_array[i, :])
        plt.title(
            f"{metric_name} for every class. Average score: {np.mean(metrics_array[i, :]):.2f}"
        )
        plt.ylabel(metric_name)
        plt.xlabel("Cell Type")
        plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, "metrics_per_class_comparison.png"))
    else:
        plt.show()


def plot_scatter_metrics(
    metric1: np.ndarray,
    metric2: np.ndarray,
    metric1_name: str,
    metric2_name: str,
    classnames: list,
    output_dir: str = None,
):
    """Compare two metrics across cell types with a scatterplot.

    Args:
        metric1 (np.ndarray): Array of metric values (1, C).
        metric2 (np.ndarray): Array of metric values (1, C).
        metric1_name (str): Name of the first metric (used for labeling).
        metric2_name (str): Name of the second metric (used for labeling).
        classnames (list): List of cell type names.
        output_dir (str, optional): Directory to save plots. None if showing.
    """
    if metric1.shape != metric2.shape:
        raise ValueError("Metrics arrays must have the same shape")

    if len(metric1) != len(classnames):
        raise ValueError("Length of metrics arrays must match number of cell types")

    plt.figure(figsize=(10, 8))

    for i, classname in enumerate(classnames):
        plt.scatter(metric1[i], metric2[i], label=classname, color="tab:blue")
        plt.text(metric1[i], metric2[i], classname)

    plt.xlabel(metric1_name)
    plt.ylabel(metric2_name)
    plt.title(f"Scatter Plot: {metric1_name} vs {metric2_name}")
    plt.grid(True)

    # Save or show the plot
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(
            os.path.join(output_dir, f"{metric1_name}_vs_{metric2_name}_scatter.png")
        )
    else:
        plt.show()


if __name__ == "__main__":
    # Test plotting functions
    predictions = np.random.rand(20, 19)
    groundtruth = np.random.rand(20, 19)
    metric_array = np.random.rand(4, 19)
    metric_names = ["pearson", "accuracy", "precision", "recall"]
    classnames = [f"Type {j+1}" for j in range(19)]

    plot_predictions_vs_groundtruth(
        predictions, groundtruth[0:1, :], classnames, output_dir="."
    )
    plot_metrics(metric_array, metric_names, classnames, output_dir=".")
    plot_scatter_metrics(
        metric_array[0, :],
        metric_array[1, :],
        metric_names[0],
        metric_names[1],
        classnames,
        output_dir=".",
    )
    plot_scatter_predictions_vs_groundtruths(
        predictions,
        groundtruth,
        classnames,
        output_dir=".",
    )
