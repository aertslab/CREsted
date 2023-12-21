"""Load a trained model, predict on the test set and show performance scores."""
import argparse
import os
import pandas as pd
import pyfaidx
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabulate import tabulate
from scipy import stats
from utils.one_hot_encoding import get_hot_encoding_table, regions_to_hot_encoding
from utils.plot import plot_scatter_predictions_vs_groundtruths, plot_metrics


def parse_arguments() -> argparse.Namespace:
    """Parse required command line arguments."""
    parser = argparse.ArgumentParser(description="Predict on a model.")
    parser.add_argument(
        "-g",
        "--genome_fasta_file",
        type=str,
        help="Path to genome FASTA file",
        default="data/raw/genome.fa",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to output model directory (e.g. 'checkpoints/mouse/2023-12-20_15:32/')",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Trained .keras model name inside model_dir/checkpoints. \
        If empty, will load last model in directory.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    # Load test data
    targets = np.load(os.path.join(args.model_dir, "targets.npz"))["test"]
    one_hot_encoding_table = get_hot_encoding_table()
    genomic_pyfasta = pyfaidx.Fasta(args.genome_fasta_file, sequence_always_upper=True)
    regions_bed_file = os.path.join(args.model_dir, "regions.bed")
    region_split_ids = np.load(os.path.join(args.model_dir, "region_split_ids.npz"))

    classnames = []
    with open(
        os.path.join(args.model_dir, "cell_type_mapping.tsv"), "r"
    ) as cell_mapping:
        for line in cell_mapping:
            classnames.append(line.strip().split("\t")[1])

    # Read regions and one hot encode test regions
    seqs_one_hot = regions_to_hot_encoding(
        regions_bed_filename=regions_bed_file,
        genomic_pyfasta=genomic_pyfasta,
        hot_encoding_table=one_hot_encoding_table,
        idx=region_split_ids["test"],
    )

    # Load model
    if args.model_name:
        model = tf.keras.models.load_model(
            os.path.join(args.model_dir, "checkpoints", args.model_name), compile=False
        )
    else:
        model_checkpoints = os.listdir(os.path.join(args.model_dir, "checkpoints"))
        model_checkpoints.sort()
        last_model = model_checkpoints[-1]
        print(f"No model selected. Loading last model {last_model} in directory...")
        model = tf.keras.models.load_model(
            os.path.join(args.model_dir, "checkpoints", last_model), compile=False
        )
    print("Model loaded successfully")

    # Predict
    print("Predicting on test set...")
    test_predictions = model.predict(seqs_one_hot)
    del seqs_one_hot

    # Calculate performance metrics accross cell types
    metrics_data = {
        "Cell Type": [],
        "MAE": [],
        "MSE": [],
        "Pearson": [],
        "Spearman": [],
    }
    for i, cell_type in enumerate(classnames):
        mae = mean_absolute_error(targets[:, i], test_predictions[:, i])
        mse = mean_squared_error(targets[:, i], test_predictions[:, i])
        pearson_corr, _ = stats.pearsonr(targets[:, i], test_predictions[:, i])
        spearman_corr, _ = stats.spearmanr(targets[:, i], test_predictions[:, i])

        metrics_data["Cell Type"].append(cell_type)
        metrics_data["MAE"].append(mae)
        metrics_data["MSE"].append(mse)
        metrics_data["Pearson"].append(pearson_corr)
        metrics_data["Spearman"].append(spearman_corr)

    metrics_df = pd.DataFrame(metrics_data)
    print("\nMetrics per Cell Type:")
    print(tabulate(metrics_df, headers="keys", tablefmt="psql"))

    metrics_df = pd.DataFrame(metrics_data)
    avg_metrics_df = pd.DataFrame(
        {
            "Average Metrics": ["MAE", "MSE", "Pearson", "Spearman"],
            "Value": [
                metrics_df["MAE"].mean(),
                metrics_df["MSE"].mean(),
                metrics_df["Pearson"].mean(),
                metrics_df["Spearman"].mean(),
            ],
        }
    )

    print("\nAverage Metrics:")
    print(tabulate(avg_metrics_df, headers="keys", tablefmt="psql"))

    # Save metrics
    output_folder = os.path.join(args.model_dir, "evaluation_outputs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metrics_df.to_csv(
        os.path.join(output_folder, "metrics_per_cell_type.tsv"), sep="\t", index=False
    )
    avg_metrics_df.to_csv(
        os.path.join(output_folder, "metrics_average.tsv"), sep="\t", index=False
    )

    print(f"Metrics tsv's saved to {output_folder}/.")

    # Create metric & prediction plots & save
    metric_array = metrics_df.iloc[:, 1:5].values
    metric_array = np.transpose(metric_array)
    metric_names = ["MAE", "MSE", "Pearson", "Spearman"]

    plot_metrics(metric_array, metric_names, classnames, output_dir=output_folder)
    plot_scatter_predictions_vs_groundtruths(
        test_predictions, targets, classnames, output_dir=output_folder
    )
    print(f"Plots saved to {output_folder}/.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
