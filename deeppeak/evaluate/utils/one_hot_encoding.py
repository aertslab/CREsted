import pyfaidx
import numpy as np
from tqdm import tqdm


def regions_to_hot_encoding(
    regions_bed_filename: str,
    genomic_pyfasta: pyfaidx.Fasta,
    hot_encoding_table: np.ndarray,
):
    """
    Encode the seqeunce associated with each region in regions_bed_filename
    to a hot encoded numpy array with shape (len(sequence), len(alphabet)).
    """

    def _get_region_from_bed(regions_bed_filename: str):
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

    # Get a region (chrom, start, end) from the regions BED file.
    for region in _get_region_from_bed(regions_bed_filename):
        # Region is in BED format: zero-based half open interval.
        chrom, start, end = region
        sequence = str(genomic_pyfasta[chrom][start:end].seq)
        # Hot encode region.
        sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        yield hot_encoding_table[sequence_bytes]


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
