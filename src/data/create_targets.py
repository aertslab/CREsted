"""Create target vectors from the preprocessed bigwig files."""

import os
import numpy as np
import click
import logging
from tqdm import tqdm


@click.command()
@click.argument("input_folder", type=str)
@click.argument("output_folder", type=str)
def main(input_folder: str, output_folder: str):
    print("\nCreating target vectors...")
    # Directory containing preprocessed bigwig TSV files (from script)
    tsv_dir = os.path.join(input_folder, "bw")
    tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith(".tsv")]

    num_cell_types = len(tsv_files)
    with open(os.path.join(tsv_dir, tsv_files[0]), "r") as f:
        num_regions = len(f.readlines())  # Number of regions in each TSV file

    # Create target vector
    print(f"\nCreating target vector from {tsv_dir}...")
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
    print(f"Saving target vector to {output_folder}...")
    np.save(os.path.join(output_folder, "targets.npy"), target_vector)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
