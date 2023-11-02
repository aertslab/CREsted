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
import tensorflow as tf
import shutil


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_data_as_tfrecord(
    prefix, X_data, Y_data, output_folder, samples_per_file=10000
):
    """
    Save data as TFRecords, chunked by samples_per_file.
    """
    num_samples = X_data.shape[0]
    num_files = (num_samples + samples_per_file - 1) // samples_per_file

    if not os.path.exists(os.path.join(output_folder, prefix)):
        os.makedirs(os.path.join(output_folder, prefix))
    else:
        print(f"WARNING: Removing {os.path.join(output_folder, prefix)}")
        shutil.rmtree(os.path.join(output_folder, prefix))
        os.makedirs(os.path.join(output_folder, prefix))

    for idx in range(num_files):
        start_idx = idx * samples_per_file
        end_idx = min((idx + 1) * samples_per_file, num_samples)

        file_name = f"{prefix}_{idx}.tfrecord"
        file_path = os.path.join(output_folder, prefix, file_name)

        with tf.io.TFRecordWriter(file_path) as writer:
            for X, Y in zip(X_data[start_idx:end_idx], Y_data[start_idx:end_idx]):
                example = serialize_example(X, Y)
                writer.write(example)


def serialize_example(X, Y):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Convert the arrays to bytes after converting to uint8
    X_bytes = tf.io.serialize_tensor(tf.cast(X, tf.uint8)).numpy()
    Y_bytes = tf.io.serialize_tensor(Y).numpy()

    feature = {
        "X": _bytes_feature(X_bytes),
        "Y": _bytes_feature(Y_bytes),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


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
    print("\nSplitting data into train, validation, and test sets...")
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

    # # Augment X_train by taking inverse complement bases
    # X_train = np.concatenate([X_train, X_train[:, ::-1, ::-1]], axis=0)
    # Y_train = np.concatenate([Y_train, Y_train], axis=0)

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print("Y_train:", Y_train.shape)
    print("Y_val:", Y_val.shape)
    print("Y_test:", Y_test.shape)

    # Save
    # np.save(os.path.join(output_folder, "X_train.npy"), X_train)
    # np.save(os.path.join(output_folder, "X_val.npy"), X_val)
    # np.save(os.path.join(output_folder, "X_test.npy"), X_test)

    # np.save(os.path.join(output_folder, "Y_train.npy"), Y_train)
    # np.save(os.path.join(output_folder, "Y_val.npy"), Y_val)
    # np.save(os.path.join(output_folder, "Y_test.npy"), Y_test)
    # Save
    print("Saving data as TFRecords...")

    save_data_as_tfrecord("train", X_train, Y_train, output_folder)
    save_data_as_tfrecord("val", X_val, Y_val, output_folder)
    save_data_as_tfrecord("test", X_test, Y_test, output_folder)


if __name__ == "__main__":
    main()
