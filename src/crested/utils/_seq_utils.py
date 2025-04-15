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

    # 256 x 4
    hot_encoding_table = np.zeros(
        (np.iinfo(np.uint8).max + 1, len(alphabet)), dtype=dtype
    )

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


def one_hot_encode_sequence(sequence: str, expand_dim: bool = True) -> np.ndarray:
    """
    One hot encode a DNA sequence.

    Will return a numpy array with shape (len(sequence), 4) if expand_dim is True, otherwise (4,).
    Alphabet is ACGT.

    Parameters
    ----------
    sequence
        The DNA sequence to one hot encode.
    expand_dim
        Whether to expand the dimensions of the output array.

    Returns
    -------
    The one hot encoded DNA sequence.
    """
    if expand_dim:
        return np.expand_dims(
            HOT_ENCODING_TABLE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)],
            axis=0,
        )
    else:
        return HOT_ENCODING_TABLE[
            np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        ]


def generate_mutagenesis(x, include_original=True, flanks=(0, 0)):
    """Generate all possible single point mutations in a sequence."""
    _, L, A = x.shape
    start, end = 0, L
    x_mut = []
    start = flanks[0]
    end = L - flanks[1]
    for length in range(start, end):
        for a in range(A):
            if not include_original:
                if x[0, length, a] == 1:
                    continue
            x_new = np.copy(x)
            x_new[0, length, :] = 0
            x_new[0, length, a] = 1
            x_mut.append(x_new)
    return np.concatenate(x_mut, axis=0)


def generate_motif_insertions(x, motif, flanks=(0, 0), masked_locations=None):
    """Generate motif insertions in a sequence."""
    _, L, A = x.shape
    start, end = 0, L
    x_mut = []
    motif_length = motif.shape[1]
    start = flanks[0]
    end = L - flanks[1] - motif_length + 1
    insertion_locations = []

    for motif_start in range(start, end):
        motif_end = motif_start + motif_length
        if masked_locations is not None:
            if np.any(
                (motif_start <= masked_locations) & (masked_locations < motif_end)
            ):
                continue
        x_new = np.copy(x)
        x_new[0, motif_start:motif_end, :] = motif
        x_mut.append(x_new)
        insertion_locations.append(motif_start)

    return np.concatenate(x_mut, axis=0), insertion_locations


def generate_window_shuffle(
    x, window_size, n_shuffles, uniform, include_original=True, flanks=(0, 0)
):
    """Generate all possible single point mutations in a sequence."""
    _, L, A = x.shape
    start, end = 0, L
    x_mut = []
    start = flanks[0]
    end = L - flanks[1] - window_size + 1
    rng = np.random.default_rng()
    uniform_array = np.identity(4, dtype=int)
    for _ in range(n_shuffles):
        for location in range(start, end):
            x_new = np.copy(x)
            if uniform:
                x_new[0, location : location + window_size, :] = rng.choice(
                    uniform_array, window_size
                )
            else:
                rng.shuffle(x_new[0, location : location + window_size, :], axis=0)
            x_mut.append(x_new)
    return np.concatenate(x_mut, axis=0)


def build_one_hot_decoding_table() -> np.ndarray:
    """Get hot decoding table to decode a one hot encoded sequence to a DNA sequence string."""
    one_hot_decoding_table = np.full(
        np.iinfo(np.uint8).max + 1, ord("N"), dtype=np.uint8
    )
    one_hot_decoding_table[1] = ord("A")
    one_hot_decoding_table[2] = ord("C")
    one_hot_decoding_table[4] = ord("G")
    one_hot_decoding_table[8] = ord("T")

    return one_hot_decoding_table


HOT_DECODING_TABLE = build_one_hot_decoding_table()


def hot_encoding_to_sequence(one_hot_encoded_sequence: np.ndarray) -> str:
    """
    Decode a one hot encoded sequence to a DNA sequence string.

    Parameters
    ----------
    one_hot_encoded_sequence
        A numpy array with shape (x, 4) with dtype=np.float32.

    Returns
    -------
    The DNA sequence string of length x.
    """
    # Convert hot encoded seqeuence from:
    #   (x, 4) with dtype=np.float32
    # to:
    #   (x, 4) with dtype=np.uint8
    # and finally combine ACGT dimensions to:
    #   (x, 1) with dtype=np.uint32
    hes_u32 = one_hot_encoded_sequence.astype(np.uint8).view(np.uint32)

    # Do some bitshifting magic to decode uint32 to DNA sequence string.
    sequence = (
        HOT_DECODING_TABLE[
            (
                (
                    hes_u32 << 31 >> 31
                )  # A: 2^0  : 1        -> 1 = A in HOT_DECODING_TABLE
                | (
                    hes_u32 << 23 >> 30
                )  # C: 2^8  : 256      -> 2 = C in HOT_DECODING_TABLE
                | (
                    hes_u32 << 15 >> 29
                )  # G: 2^16 : 65536    -> 4 = G in HOT_DECODING_TABLE
                | (
                    hes_u32 << 7 >> 28
                )  # T: 2^24 : 16777216 -> 8 = T in HOT_DECODING_TABLE
            ).astype(np.uint8)
        ]
        .tobytes()
        .decode("ascii")
    )

    return sequence


def reverse_complement(sequence: str | list[str] | np.ndarray) -> str | np.ndarray:
    """
    Perform reverse complement on either a one-hot encoded array or a (list of) DNA sequence string(s).

    Parameters
    ----------
    sequence
        The DNA sequence string(s) or one-hot encoded array to reverse complement.

    Returns
    -------
    The reverse complemented DNA sequence string or one-hot encoded array.
    """

    def complement_str(seq: str) -> str:
        complement = str.maketrans("ACGTacgt", "TGCAtgca")
        return seq.translate(complement)[::-1]

    if isinstance(sequence, str):
        return complement_str(sequence)
    elif isinstance(sequence, list):
        return [complement_str(seq) for seq in sequence]
    elif isinstance(sequence, np.ndarray):
        if sequence.ndim == 2:
            if sequence.shape[1] == 4:
                return sequence[::-1, ::-1]
            elif sequence.shape[0] == 4:
                return sequence[:, ::-1][:, ::-1]
            else:
                raise ValueError(
                    "One-hot encoded array must have shape (W, 4) or (4, W)"
                )
        elif sequence.ndim == 3:
            if sequence.shape[1] == 4:
                return sequence[:, ::-1, ::-1]
            elif sequence.shape[2] == 4:
                return sequence[:, ::-1, ::-1]
            else:
                raise ValueError(
                    "One-hot encoded array must have shape (B, 4, W) or (B, W, 4)"
                )
        else:
            raise ValueError("One-hot encoded array must have 2 or 3 dimensions")
    else:
        raise TypeError(
            "Input must be either a DNA sequence string or a one-hot encoded array"
        )
