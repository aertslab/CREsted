"""Create target vectors from the preprocessed bigwigs."""

import os
import argparse
import numpy as np
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create target vectors from the preprocessed bigwig files."
    )
    parser.add_argument(
        "-b",
        "--bigwig_dir",
        type=str,
        help="Path to the folder containing the preprocessed bigwig files.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the folder to save the target vectors.",
        required=True,
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    print("\nCreating target vectors...")
    # Directory containing preprocessed bigwig TSV files (from script)
    tsv_dir = args.bigwig_dir
    tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith(".tsv")]
    tsv_files.sort()

    num_cell_types = len(tsv_files)
    with open(os.path.join(tsv_dir, tsv_files[0]), "r") as f:
        num_regions = len(f.readlines())  # Number of regions in each TSV file

    # Create target vector
    print(f"Creating target vectors from {tsv_dir}...")
    target_vector = np.zeros((2, num_regions, num_cell_types))

    for cell_type_idx, tsv_file in tqdm(enumerate(tsv_files), total=num_cell_types):
        file_path = os.path.join(tsv_dir, tsv_file)
        with open(file_path, "r") as f:
            for region_idx, line in enumerate(f):
                columns = line.strip().split("\t")
                average_peak_height = float(columns[-4])
                max_peak_height = float(columns[-1])
                target_vector[0, region_idx, cell_type_idx] = max_peak_height
                target_vector[1, region_idx, cell_type_idx] = average_peak_height

    # Save target vector
    print(f"Saving target vector to {args.output_dir}...")
    np.save(os.path.join(args.output_dir, "targets.npy"), target_vector)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
