"""Helper functions for loading data from tfRecords."""
import tensorflow as tf
from augment import complement_base
from functools import partial
import pyfaidx
import numpy as np


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


def get_hot_encoding_table(
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    """
    Get hot encoding table to encode a DNA sequence to a numpy array with shape
    (len(sequence), len(alphabet)) using bytes.
    """

    def str_to_uint8(string) -> np.ndarray:
        """
        Convert string to byte representation.
        """
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    # 255 x 4
    hot_encoding_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)

    # For each ASCII value of the nucleotides used in the alphabet
    # (upper and lower case), set 1 in the correct column.
    hot_encoding_table[str_to_uint8(alphabet.upper())] = np.eye(
        len(alphabet), dtype=dtype
    )
    hot_encoding_table[str_to_uint8(alphabet.lower())] = np.eye(
        len(alphabet), dtype=dtype
    )

    # For each ASCII value of the nucleotides used in the neutral alphabet
    # (upper and lower case), set neutral_value in the correct column.
    hot_encoding_table[str_to_uint8(neutral_alphabet.upper())] = neutral_value
    hot_encoding_table[str_to_uint8(neutral_alphabet.lower())] = neutral_value

    return hot_encoding_table


def get_regions_from_bed(regions_bed_filename: str):
    """
    Read BED file and yield a region (chrom, start, end) for each invocation.
    """
    with open(regions_bed_filename, "r") as fh_bed:
        for line in fh_bed:
            line = line.rstrip("\r\n")

            if line.startswith("#"):
                continue

            columns = line.split("\t")
            chrom = columns[0]
            start, end = [int(x) for x in columns[1:3]]
            region = chrom, start, end
            yield region


class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(
        self,
        bed_file: str,
        genome_fasta_file: str,
        targets: str,
        split_dict: dict,
        set_type: str,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.bed_file = bed_file
        self.genome_fasta_file = genome_fasta_file
        self.targets = np.load(targets)
        self.split_dict = split_dict
        self.set_type = set_type
        self.batch_size = batch_size
        self.genomic_pyfasta = pyfaidx.Fasta(
            genome_fasta_file, sequence_always_upper=True
        )
        self.hot_encoding_table = get_hot_encoding_table()
        self.all_regions = list(get_regions_from_bed(bed_file))

        self.indices = self._get_indices_for_set_type(set_type)

        self.shuffle = shuffle

    def _get_indices_for_set_type(self, set_type: str):
        """Get indices for the specified set type (train/val/test)."""
        if set_type not in ["train", "val", "test"]:
            raise ValueError("set_type must be 'train', 'val', or 'test'")

        if set_type == "train":
            excluded_chromosomes = set(
                self.split_dict.get("val", []) + self.split_dict.get("test", [])
            )
            all_chromosomes = set(chrom for chrom, _, _ in self.all_regions)
            selected_chromosomes = all_chromosomes - excluded_chromosomes
        else:
            selected_chromosomes = set(self.split_dict.get(set_type, []))

        indices = [
            i
            for i, region in enumerate(self.all_regions)
            if region[0] in selected_chromosomes
        ]
        return indices

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        """Return one batch of data."""
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x = np.zeros((len(batch_indices), 2114, 4), dtype=np.float32)
        batch_y = np.array([self.targets[1, i] for i in batch_indices])

        for i, index in enumerate(batch_indices):
            chrom, start, end = self.all_regions[index]
            sequence = str(self.genomic_pyfasta[chrom][start:end].seq)
            sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
            batch_x[i] = self.hot_encoding_table[sequence_bytes]

        return batch_x, batch_y

    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == "__main__":
    # test dataloader
    bed_file = "data/interim/consensus_peaks_2114.bed"
    genome_fasta_file = "data/raw/genome.fa"
    targets = "data/interim/targets.npy"

    split_dict = {"val": ["chr8", "chr10"], "test": ["chr9", "chr18"]}
    set_type = "train"
    batch_size = 1024

    loader = CustomDataLoader(
        bed_file, genome_fasta_file, targets, split_dict, set_type, batch_size, True
    )
    print(len(loader))
    # Get one batch
    for x, y in loader:
        print(x.shape, y.shape)
    print("finished")
