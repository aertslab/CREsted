"""Create target vectors from the preprocessed bigwigs."""

import os
import yaml
import argparse
import numpy as np
from typing import Tuple
from tqdm import tqdm
from helpers.bed import filter_bed_on_idx


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


def calc_gini(targets: np.ndarray) -> np.ndarray:
    """Returns gini scores for the given targets"""

    def _gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        array = (
            array.flatten().clip(0, None) + 0.0000001
        )  # Ensure non-negative values and avoid zero
        array = np.sort(array)
        index = np.arange(1, array.size + 1)
        return (np.sum((2 * index - array.size - 1) * array)) / (
            array.size * np.sum(array)
        )

    gini_scores = np.zeros_like(targets)

    for region_idx in range(targets.shape[0]):
        region_scores = targets[region_idx]
        max_idx = np.argmax(region_scores)
        gini_scores[region_idx, max_idx] = _gini(region_scores)

    return gini_scores


def normalize_peaks(
    target_vector: np.ndarray,
    bws: list,
    num_cell_types: int,
    threshold: int = 0,
    gini_threshold: float = 0.25,
    top_k_percent: float = 0.01,
) -> np.ndarray:
    """
    Normalize the given target vector based on top Gini coefficients.

    Calculates gini scores for the top_k highest peaks. Gini scores
    below gini_threshold are considered 'high' (in variability) and are used to
    calculate weights per cell type, which are then used to normalize the targets
    accross cells types.

    Parameters:
    - target_vector (np.ndarray): The target vector to be normalized.
    - bws (list): A list of BigWig files corresponding to each cell type.
    - num_cell_types (int): The number of cell types in the target vector.
    - threshold (int): A threshold value for filtering the target vector.
    - gini_threshold (float): Threshold for Gini scores to identify high variability.
    - top_k_percent (float): The percentage (expressed as a fraction) of top values to
      consider for Gini score calculation.

    Returns:
    - np.ndarray: The normalized target vector with adjustments based on Gini score.
    """
    top_k_percent_means = []
    gini_scores_all = []

    print("Filtering on top k Gini scores...")
    for i in range(num_cell_types):
        filtered_col = target_vector[1][:, i][target_vector[1][:, i] > threshold]
        sorted_col = np.sort(filtered_col)[::-1]
        top_k_index = int(len(sorted_col) * top_k_percent)

        gini_scores = calc_gini(
            target_vector[1, np.argsort(filtered_col)[::-1][:top_k_index]]
        )
        high_gini_indices = np.where(np.max(gini_scores, axis=1) < gini_threshold)[0]
        print(f"{len(high_gini_indices)} out of {top_k_index} remain for {bws[i]}.")

        if len(high_gini_indices) > 0:
            top_k_mean = np.mean(sorted_col[high_gini_indices])
            gini_scores_all.append(np.max(gini_scores[high_gini_indices], axis=1))
        else:
            top_k_mean = 0
            gini_scores_all.append(0)

        top_k_percent_means.append(top_k_mean)

    max_mean = np.max(top_k_percent_means)
    weights = max_mean / np.array(top_k_percent_means)
    print("Cell type weights:", weights)

    for j in range(3):
        target_vector[j] = target_vector[j] * weights

    target_vector[3] = np.log(target_vector[2] + 1)
    return target_vector


def filter_regions_on_specificity(
    target_vector: np.ndarray,
    bed_filename: str,
    gini_std_threshold: float = 1,
    target_idx: int = 1,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Filter bed regions & targets based on high Gini score. Saves filtered bed regions
    back to original file and returns filtered targets.

    Args:
        target_vector (np.ndarray): targets
        bed_filename (str): path to BED file
        gini_threshold (float): Threshold for Gini scores to identify high variability.
        target_idx (int): Type of targets to use for filtering decision (1='mean')
    """

    gini_scores = calc_gini(target_vector[target_idx])
    mean = np.mean(np.max(gini_scores, axis=1))
    std_dev = np.std(np.max(gini_scores, axis=1))
    gini_threshold = mean + gini_std_threshold * std_dev
    selected_indices = np.argwhere(np.max(gini_scores, axis=1) > gini_threshold)[:, 0]

    target_vector_filt = target_vector[:, selected_indices]
    filter_bed_on_idx(bed_filename, selected_indices)
    print(
        f"Kept {len(target_vector_filt[0,:])} out of {len(target_vector[0,:])} regions."
    )

    return target_vector_filt


def main(args: argparse.Namespace, config: dict):
    # Directory containing preprocessed bigwig TSV files (from script)
    tsv_dir = args.bigwig_dir
    tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith(".tsv")]
    tsv_files.sort()

    num_cell_types = len(tsv_files)
    with open(os.path.join(tsv_dir, tsv_files[0]), "r") as f:
        num_regions = len(f.readlines())  # Number of regions in each TSV file

    # Create target vector
    print(f"Creating target vectors from {tsv_dir}...")
    target_vector = np.zeros((4, num_regions, num_cell_types))

    for cell_type_idx, tsv_file in tqdm(enumerate(tsv_files), total=num_cell_types):
        file_path = os.path.join(tsv_dir, tsv_file)
        with open(file_path, "r") as f:
            for region_idx, line in enumerate(f):
                columns = line.strip().split("\t")
                average_peak_height = float(columns[-4])
                max_peak_height = float(columns[-1])
                count = float(columns[-5])
                target_vector[0, region_idx, cell_type_idx] = max_peak_height
                target_vector[1, region_idx, cell_type_idx] = average_peak_height
                target_vector[2, region_idx, cell_type_idx] = count
                target_vector[3, region_idx, cell_type_idx] = np.log(count + 1)

    # Normalization & specificity filtering
    if config["gini_normalization"]:
        print("Normalizing peaks...")
        target_vector = normalize_peaks(target_vector, tsv_files, num_cell_types)

    if config["specificity_filtering"]:
        print("Filtering regions based on region specificity...")
        target_vector = filter_regions_on_specificity(
            target_vector, args.regions_bed_file
        )

    if config["shift_augmentation"]["use"]:
        print("Warning: extending target matrix since shift augmentation was used.")
        total_rows_per_region = int(config["shift_augmentation"]["n_shifts"]) * 2 + 1
        binary_matrix_np = np.repeat(
            binary_matrix_np, repeats=total_rows_per_region, axis=1
        )

    # Save target vector
    print(f"Saving deeppeak target vectors to {args.output_dir}targets_deeppeak.npz...")
    np.savez_compressed(
        os.path.join(args.output_dir, "targets_deeppeak.npz"),
        targets=target_vector,
    )

    # Save cell type mapping file
    print(f"Saving cell type mapping to {args.output_dir}cell_type_mapping.tsv...")
    with open(os.path.join(args.output_dir, "cell_type_mapping.tsv"), "w") as f:
        for cell_type_idx, tsv_file in enumerate(tsv_files):
            out_path = os.path.join(args.bigwig_dir, tsv_file)
            f.write(f"{cell_type_idx}\t{tsv_file.split('.')[0]}\t{out_path}\n")


if __name__ == "__main__":
    args = parse_arguments()
    assert os.path.exists(
        "configs/user.yml"
    ), "users.yml file not found. Please run `make copyconfig` first"
    with open("configs/user.yml", "r") as f:
        config = yaml.safe_load(f)
    main(args, config)
