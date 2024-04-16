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

    # Convert to numpy array (topics x regions matrix)
    binary_matrix_np = binary_matrix.to_numpy()

    # Save the binary matrix using numpy
    print(
        f"Saving deeptopic target vectors to {args.output_dir}targets_deeptopic.npz..."
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "targets_deeptopic.npz"),
        targets=binary_matrix_np,
    )


if __name__ == "__main__":
    args = parse_arguments()
    assert os.path.exists(
        "configs/user.yml"
    ), "users.yml file not found. Please run `make copyconfig` first"
    with open("configs/user.yml", "r") as f:
        config = yaml.safe_load(f)
    main(args, config)
