"""
Using preprocessed genomic sequences (using beds) and bigwigs;
create inputs and targets for the model and split into train, validation and test sets.
"""

import click
import yaml
import os
from pybedtools import BedTool
import numpy as np
import gc


def split_indices(df, val_chr, test_chr):
    """Split the dataframe into train, validation, and test."""

    # Create conditions for validation and test sets
    val_condition = df["chrom"].isin(val_chr)
    test_condition = df["chrom"].isin(test_chr)

    # Derive indices for train, val, and test sets
    train_idcs = df.index[~(val_condition | test_condition)].tolist()
    val_idcs = df.index[val_condition].tolist()
    test_idcs = df.index[test_condition].tolist()

    return train_idcs, val_idcs, test_idcs


@click.command()
@click.argument("input_folder", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path(exists=True))
def main(input_folder: str, output_folder: str):
    print("Splitting data into train, validation, and test sets...")
    # Get val & test chr names
    with open("configs/user.yml", "r") as f:
        config = yaml.safe_load(f)
    val_chr = config["val"]  # ['chr8', 'chr10']
    test_chr = config["test"]  # ['chr9', 'chr18']

    # Load data
    bed = BedTool(os.path.join(input_folder, "consensus_peaks_2114.bed"))
    df = bed.to_dataframe()
    inputs = np.load(os.path.join(input_folder, "peaks_one_hot.npy"), mmap_mode="r")
    targets = np.load(os.path.join(input_folder, "targets.npy"), mmap_mode="r")

    # Split
    train_idcs, val_idcs, test_idcs = split_indices(df, val_chr, test_chr)

    del bed, df
    gc.collect()

    Y_train = targets[1, train_idcs]
    Y_val = targets[1, val_idcs]
    Y_test = targets[1, test_idcs]

    X_train = inputs[train_idcs]
    X_val = inputs[val_idcs]
    X_test = inputs[test_idcs]

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print("Y_train:", Y_train.shape)
    print("Y_val:", Y_val.shape)
    print("Y_test:", Y_test.shape)

    # Save
    print("Saving data...")
    np.save(os.path.join(output_folder, "X_train.npy"), X_train)
    np.save(os.path.join(output_folder, "X_val.npy"), X_val)
    np.save(os.path.join(output_folder, "X_test.npy"), X_test)

    np.save(os.path.join(output_folder, "Y_train.npy"), Y_train)
    np.save(os.path.join(output_folder, "Y_val.npy"), Y_val)
    np.save(os.path.join(output_folder, "Y_test.npy"), Y_test)


if __name__ == "__main__":
    main()
