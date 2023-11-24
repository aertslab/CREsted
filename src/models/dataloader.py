"""Helper functions for loading data from tfRecords."""
import tensorflow as tf
from augment import complement_base
import pyfaidx
import numpy as np


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
        augment_complement: bool = True,
        batch_size: int = 32,
        shuffle: bool = True,
        fraction_of_data: float = 1.0,
    ):
        self.bed_file = bed_file
        self.genome_fasta_file = genome_fasta_file
        self.targets = np.load(targets)
        self.split_dict = split_dict
        self.set_type = set_type
        self.augment_complement = augment_complement
        self.batch_size = batch_size
        self.genomic_pyfasta = pyfaidx.Fasta(
            genome_fasta_file, sequence_always_upper=True
        )
        self.hot_encoding_table = get_hot_encoding_table()
        self.all_regions = list(get_regions_from_bed(bed_file))

        self.indices = self._get_indices_for_set_type(set_type)

        self.shuffle = shuffle
        self.fraction_of_data = fraction_of_data

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
        if self.fraction_of_data < 1.0:
            return int(
                np.ceil(len(self.indices) * self.fraction_of_data / self.batch_size)
            )
        else:
            return int(np.ceil(len(self.indices) / self.batch_size))

    def __num_samples__(self):
        if self.fraction_of_data < 1.0:
            return int(np.ceil(len(self.indices) * self.fraction_of_data))
        else:
            return len(self.indices)

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
            if self.augment_complement:
                batch_x[i] = complement_base(batch_x[i])

        return np.array(batch_x), np.array(batch_y)

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
    set_type = "val"
    batch_size = 1

    loader = CustomDataLoader(
        bed_file,
        genome_fasta_file,
        targets,
        split_dict,
        set_type,
        True,
        batch_size,
        False,
    )
    print(len(loader))
    # Get one batch
    for x, y in loader:
        print(x.shape, y.shape)
        break
    print("finished")
