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


def load_chunked_tfrecord_dataset(file_pattern, config, num_samples, batch_size):
    """
    Load data from multiple chunked TFRecord files and shuffle the data.
    """
    # List files and shuffle them
    files = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    dataset = tf.data.TFRecordDataset(files)
    parse_func = partial(_parse_function, file_pattern=file_pattern, config=config)

    dataset = dataset.shuffle(buffer_size=config["shuffle_buffer_size"])
    # dataset = dataset.repeat(config["epochs"])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(parse_func)

    if config["fraction_of_data"] < 1.0:
        dataset = dataset.take(int(num_samples))
    dataset = dataset.repeat(config["epochs"])
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch
    return dataset


def _parse_function(example_proto, file_pattern: str, config: dict):
    """Load a batch of examples from a TFRecord file and augment the data."""
    feature_description = {
        "X": tf.io.FixedLenFeature([], tf.string),
        "Y": tf.io.FixedLenFeature([], tf.string),
    }
    # Parse all examples in the batch
    examples = tf.io.parse_example(example_proto, feature_description)

    # Decode each example in the batch
    X = tf.map_fn(
        lambda x: tf.io.parse_tensor(x, out_type=tf.uint8),
        examples["X"],
        dtype=tf.uint8,
    )
    Y = tf.map_fn(
        lambda x: tf.io.parse_tensor(x, out_type=tf.double),
        examples["Y"],
        dtype=tf.double,
    )

    # Cast it back to float
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)

    # Data augmentations here
    if "train" in file_pattern:
        if config["augment_complement"]:
            X = tf.map_fn(complement_base, X, dtype=tf.float32)

    # Set the shape of the tensors in the batch
    X.set_shape([None, config["seq_len"], 4])
    Y.set_shape([None, config["num_classes"]])

    return X, Y
