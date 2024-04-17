import argparse
import yaml
import os
from pathlib import Path
import pandas as pd
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create target vectors from the preprocessed bigwig files."
    )
    parser.add_argument(
        "-t",
        "--topics_dir",
        type=str,
        help="Path to the folder containing the bed files per topic.",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--regions_bed_file",
        type=str,
        help="Path to the input regions BED file.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the folder to save the target vectors.",
        required=True,
    )
    return parser.parse_args()


def extract_topic_number(filename):
    """Extracts the integer part from the filename. Assumes format 'Topic_X.bed'."""
    return int(filename.stem.split("_")[1])


def main(args, config):
    topics_folder = Path(args.topics_dir)
    peaks_file = Path(args.regions_bed_file)

    # Input checks
    if not topics_folder.is_dir():
        raise FileNotFoundError(f"Directory '{topics_folder}' not found")
    if not peaks_file.is_file():
        raise FileNotFoundError(f"File '{peaks_file}' not found")

    # Read consensus regions BED file
    consensus_peaks = pd.read_csv(peaks_file, sep="\t", header=None, usecols=[0, 1, 2])
    consensus_peaks["region"] = (
        consensus_peaks[0].astype(str)
        + ":"
        + consensus_peaks[1].astype(str)
        + "-"
        + consensus_peaks[2].astype(str)
    )

    binary_matrix = pd.DataFrame(0, index=[], columns=consensus_peaks["region"])

    # Which topic regions are present in the consensu regions
    for topic_file in sorted(topics_folder.glob("*.bed"), key=extract_topic_number):
        topic_name = topic_file.stem
        topic_peaks = pd.read_csv(topic_file, sep="\t", header=None, usecols=[0, 1, 2])
        topic_peaks["region"] = (
            topic_peaks[0].astype(str)
            + ":"
            + topic_peaks[1].astype(str)
            + "-"
            + topic_peaks[2].astype(str)
        )

        # Create binary row for the current topic (topics x regions matrix in sorted folder order)
        binary_row = binary_matrix.columns.isin(topic_peaks["region"]).astype(int)

        binary_matrix.loc[topic_name] = binary_row

    # Convert to numpy array
    binary_matrix_np = binary_matrix.to_numpy()
    binary_matrix_np = binary_matrix_np.T  # (regions x topics)

    if config["shift_augmentation"]["use"]:
        print("Warning: extending target matrix since shift augmentation was used.")
        total_rows_per_region = int(config["shift_augmentation"]["n_shifts"]) * 2 + 1
        binary_matrix_np = np.repeat(
            binary_matrix_np, repeats=total_rows_per_region, axis=0
        )

    # Save the binary matrix using numpy
    print(
        f"Saving deeptopic target vectors to {args.output_dir}targets_deeptopic.npz..."
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "targets_deeptopic.npz"),
        targets=binary_matrix_np,
    )

    # Save cell type/topic mapping file
    topic_files = [
        f.name for f in sorted(topics_folder.glob("*.bed"), key=extract_topic_number)
    ]
    print(f"Saving topic mapping to {args.output_dir}cell_type_mapping.tsv...")
    with open(os.path.join(args.output_dir, "cell_type_mapping.tsv"), "w") as f:
        for cell_type_idx, tsv_file in enumerate(topic_files):
            out_path = os.path.join(args.topics_dir, tsv_file)
            f.write(f"{cell_type_idx}\t{str(tsv_file).split('.')[0]}\t{out_path}\n")


if __name__ == "__main__":
    args = parse_arguments()
    assert os.path.exists(
        "configs/user.yml"
    ), "users.yml file not found. Please run `make copyconfig` first"
    with open("configs/user.yml", "r") as f:
        config = yaml.safe_load(f)
    main(args, config)
