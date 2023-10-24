"""Functions for preprocessing genomic data."""
import numpy as np

def get_hot_encoding_table(
    alphabet: str = 'ACGT',
    neutral_alphabet: str = 'N',
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
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)

    # 255 x 4
    hot_encoding_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)

    # For each ASCII value of the nucleotides used in the alphabet
    # (upper and lower case), set 1 in the correct column.
    hot_encoding_table[str_to_uint8(alphabet.upper())] = np.eye(len(alphabet), dtype=dtype)
    hot_encoding_table[str_to_uint8(alphabet.lower())] = np.eye(len(alphabet), dtype=dtype)

    # For each ASCII value of the nucleotides used in the neutral alphabet
    # (upper and lower case), set neutral_value in the correct column.
    hot_encoding_table[str_to_uint8(neutral_alphabet.upper())] = neutral_value
    hot_encoding_table[str_to_uint8(neutral_alphabet.lower())] = neutral_value

    return hot_encoding_table

