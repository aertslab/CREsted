import pyfaidx
import numpy as np
from tqdm import tqdm


def regions_to_hot_encoding(
    regions_bed_filename: str,
    genomic_pyfasta: pyfaidx.Fasta,
    hot_encoding_table: np.ndarray,
    idx: np.ndarray = None,
):
    """
    Encode the seqeunce associated with each region in regions_bed_filename
    to a hot encoded numpy array with shape (len(sequence), len(alphabet)).

    Args:
        regions_bed_filename (str): Path to regions BED file.
        genomic_pyfasta (pyfaidx.Fasta): Genome FASTA object.
        hot_encoding_table (np.ndarray): One hot encoding reference table.
        idx (np.ndarray, optional): Index of regions to one hot encode. If None,
        all regions will be one hot encoded.
    """

    def _get_regions_from_bed(regions_bed_filename: str, idx: np.ndarray) -> list:
        """
        Read BED file and return a list of regions (chrom, start, end).
        """
        regions = []
        with open(regions_bed_filename, "r") as fh_bed:
            for i, line in enumerate(fh_bed):
                line = line.rstrip("\r\n")
                if line.startswith("#"):
                    continue
                if idx is not None and i not in idx:
                    continue

                columns = line.split("\t")
                chrom = columns[0]
                start, end = [int(x) for x in columns[1:3]]
                regions.append((chrom, start, end))
        return regions

    regions = _get_regions_from_bed(regions_bed_filename, idx)
    n_regions = len(regions)
    if n_regions == 0:
        raise ValueError("No regions found for this specification")
    region_width = regions[0][2] - regions[0][1]
    num_alphabets = hot_encoding_table.shape[1]

    # Initialize the array
    seq_one_hot = np.zeros((n_regions, region_width, num_alphabets))

    # Fill in the one-hot encoded sequences
    print("One hot encoding sequences...")
    for i, (chrom, start, end) in tqdm(enumerate(regions), total=n_regions):
        sequence = str(genomic_pyfasta[chrom][start:end].seq)
        sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        encoded_sequence = hot_encoding_table[sequence_bytes]

        # Adjust this part if regions have varying widths
        seq_one_hot[i, : encoded_sequence.shape[0], :] = encoded_sequence

    return seq_one_hot


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
