from __future__ import annotations

import numpy as np


def get_hot_encoding_table(
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    """Get hot encoding table to encode a DNA sequence to a numpy array with shape (len(sequence), len(alphabet)) using bytes."""

    def str_to_uint8(string) -> np.ndarray:
        """Convert string to byte representation."""
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


HOT_ENCODING_TABLE = get_hot_encoding_table()


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """One hot encode a DNA sequence."""
    return np.expand_dims(
        HOT_ENCODING_TABLE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)],
        axis=0,
    )

def generate_mutagenesis(x, include_original=True):
        _, L, A = x.shape
        x_mut = []
        for length in range(L):
            for a in range(A):
                if not include_original:
                    if x[0, length, a] == 1:
                        continue
                x_new = np.copy(x)
                x_new[0, length, :] = 0
                x_new[0, length, a] = 1
                x_mut.append(x_new)
        return np.concatenate(x_mut, axis=0)

def _weighted_difference(mutated_predictions, original_prediction, target, class_penalty_weights=None):
        n_classes = original_prediction.shape[1]
        penalty_factor = 1 / n_classes
        
        target_increase = mutated_predictions[:, target] - original_prediction[:, target]
        other_increases = mutated_predictions - original_prediction
        
        other_increases[:, target] = 0
        

        if class_penalty_weights is None:
            class_penalty_weights = np.ones(n_classes)

        score = target_increase - penalty_factor * np.sum(other_increases*class_penalty_weights, axis=1)

        return np.argmax(score)