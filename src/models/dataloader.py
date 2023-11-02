"""Helper functions for loading data from tfRecords."""
import tensorflow as tf
from augment import complement_base
from functools import partial


def count_samples_in_tfrecords(file_pattern):
    """
    Count the number of samples in a set of TFRecord files.

    Args:
        file_pattern (str): Glob pattern to match TFRecord files.

    Returns:
        int: Total number of samples.
    """
    # Create a dataset from the file pattern
    files = tf.data.Dataset.list_files(file_pattern)

    # Initialize a counter
    num_samples = 0

    # Iterate over each file and count the number of records
    for file in files:
        # Use TFRecordDataset to read the file
        dataset = tf.data.TFRecordDataset(file)

        # Count records in this file
        num_samples += sum(1 for _ in dataset)

    return num_samples


def load_chunked_tfrecord_dataset(file_pattern, config, num_samples):
    """
    Load data from multiple chunked TFRecord files and shuffle the data.
    """
    # List files and shuffle them
    files = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    dataset = tf.data.TFRecordDataset(files)
    parse_func = partial(_parse_function, file_pattern=file_pattern, config=config)
    dataset = dataset.map(parse_func)

    if config["fraction_of_data"] < 1.0:
        dataset = dataset.take(int(num_samples))

    dataset = dataset.shuffle(buffer_size=config["shuffle_buffer_size"])
    dataset = dataset.repeat(config["epochs"])
    dataset = dataset.batch(config["batch_size"])
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    return dataset


def _parse_function(example_proto, file_pattern: str, config: dict):
    """Load a single example from a TFRecord file and augment the data."""
    feature_description = {
        "X": tf.io.FixedLenFeature([], tf.string),
        "Y": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    X = tf.io.parse_tensor(example["X"], out_type=tf.uint8)
    Y = tf.io.parse_tensor(example["Y"], out_type=tf.double)

    # Cast it back to float
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)

    # Data augmentations here
    if "train" in file_pattern:
        if config["augment_complement"]:
            X = complement_base(X)

    X.set_shape([config["seq_len"], 4])
    Y.set_shape([config["num_classes"]])

    return X, Y
