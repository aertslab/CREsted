from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import pysam
from loguru import logger

from crested._io import _extract_tracks_from_bigwig


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


class EnhancerOptimizer:
    """
    Class to optimize the mutated sequence based on the original prediction.

    Can be passed as the 'enhancer_optimizer' argument to :func:`crested.tl.Crested.enhancer_design_in_silico_evolution`

    Parameters
    ----------
    optimize_func
        Function to optimize the mutated sequence based on the original prediction.

    See Also
    --------
    crested.tl.Crested.enhancer_design_in_silico_evolution
    """

    def __init__(self, optimize_func: Callable[..., int]) -> None:
        """Initialize the EnhancerOptimizer class."""
        self.optimize_func = optimize_func

    def get_best(
        self,
        mutated_predictions: np.ndarray,
        original_prediction: np.ndarray,
        target: int | np.ndarray,
        **kwargs: dict[str, Any],
    ) -> int:
        """Get the index of the best mutated sequence based on the original prediction."""
        return self.optimize_func(
            mutated_predictions, original_prediction, target, **kwargs
        )


def _weighted_difference(
    mutated_predictions: np.ndarray,
    original_prediction: np.ndarray,
    target: int,
    class_penalty_weights: np.ndarray | None = None,
):
    if len(original_prediction.shape) == 1:
        original_prediction = original_prediction[None]
    n_classes = original_prediction.shape[1]
    penalty_factor = 1 / n_classes

    target_increase = mutated_predictions[:, target] - original_prediction[:, target]
    other_increases = mutated_predictions - original_prediction

    other_increases[:, target] = 0

    if class_penalty_weights is None:
        class_penalty_weights = np.ones(n_classes)

    score = target_increase - penalty_factor * np.sum(
        other_increases * class_penalty_weights, axis=1
    )

    return np.argmax(score)


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


def get_value_from_dataframe(df: pd.DataFrame, row_name: str, column_name: str):
    """
    Retrieve a single value from a DataFrame based on the given row index and column name.

    Parameters
    ----------
    df
        The DataFrame to retrieve the value from.
    row_name
        The name of the row.
    column_name
        The name of the column.

    Returns
    -------
    The value at the specified row index and column name, or an error message if the column is not found.
    """
    try:
        # Check if the column exists in the DataFrame
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame.")

        # Retrieve the value
        value = df.loc[row_name, column_name]
        return value
    except KeyError as e:
        # Handle the case where the column is not found
        return str(e)
    except IndexError:
        # Handle the case where the row index is out of bounds
        return f"Row index is out of bounds for DataFrame with {len(df)} rows."


def extract_bigwig_values_per_bp(
    bigwig_file: os.PathLike, coordinates: list[tuple[str, int, int]]
) -> tuple[np.ndarray, list[int]]:
    """
    Extract per-base pair values from a bigWig file for the given genomic coordinates.

    Parameters
    ----------
    bigwig_file
        Path to the bigWig file.
    coordinates
        An array of tuples, each containing the chromosome name and the start and end positions of the sequence.

    Returns
    -------
    bw_values
        A numpy array of values from the bigWig file for each base pair in the specified range.
    all_midpoints
        A list of all base pair positions covered in the specified coordinates.
    """
    logger.warning(
        "extract_bigwig_values_per_bp() is deprecated. Please use crested.utils.read_bigwig_region(bw_file, (chr, start, end)) instead."
    )
    # Calculate the full range of coordinates
    min_coord = min([int(start) for _, start, _ in coordinates])
    max_coord = max([int(end) for _, _, end in coordinates])

    # Initialize the list to store values
    bw_values = []

    # Get chromosome
    chrom = coordinates[0][0]  # Assuming all coordinates are for the same chromosome

    # Extract per-base values
    bw_values, all_midpoints = read_bigwig_region(
        bigwig_file, (chrom, min_coord, max_coord), missing=0.0
    )

    return bw_values, all_midpoints


def fetch_sequences(
    regions: str | list[str], genome: os.PathLike, uppercase: bool = True
) -> list[str]:
    """
    Fetch sequences from a genome file for a list of regions using pysam.

    Regions should be formatted as "chr:start-end".

    Parameters
    ----------
    regions
        List of regions to fetch sequences for.
    genome
        Path to the genome file.
    uppercase
        If True, return sequences in uppercase.

    Returns
    -------
    List of sequence strings for each region.

    Examples
    --------
    >>> regions = ["chr1:1000000-1000100", "chr1:1000100-1000200"]
    >>> region_seqs = crested.utils.fetch_sequences(regions, genome_path)
    """
    if isinstance(regions, str):
        regions = [regions]
    fasta = pysam.FastaFile(genome)
    seqs = []
    for region in regions:
        chrom, start_end = region.split(":")
        start, end = start_end.split("-")
        seq = fasta.fetch(chrom, int(start), int(end))
        if uppercase:
            seq = seq.upper()
        seqs.append(seq)
    return seqs


def read_bigwig_region(
    bigwig_file: os.PathLike,
    coordinates: tuple[str, int, int],
    bin_size: int | None = None,
    target: str = "mean",
    missing: float = 0.0,
    oob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-base or binned pair values from a bigWig file for a set of genomic region.

    Parameters
    ----------
    bigwig_file
        Path to the bigWig file.
    coordinates
        A tuple looking like (chr, start, end).
    bin_size
        If set, the returned values are mean-binned at this resolution.
    target
        How to summarize the values per bin, when binning. Can be 'mean', 'min', or 'max'.
    missing
        Fill-in value for unreported data in valid regions. Default is 0.
    oob
        Fill-in value for out-of-bounds regions. Default is 0.

    Returns
    -------
    values
        numpy array with the values from the bigwig for the requested coordinates. Shape: [n_bp], or [n_bp//bin_size] if bin_size is specified.
    positions
        numpy array with genomic positions as integers of the values in values. Shape: [n_bp], or [n_bp//bin_size] if bin_size is specified.

    Example
    -------
    >>> anndata = crested.read_bigwig_region(
    ...     bw_file="path/to/bigwig",
    ...     coordinates=("chr1", 0, 32000),
    ...     bin_size=32,
    ...     target="mean",
    ... )
    """
    # Check for accidental passing of lists of coordinates or wrong orders
    if not (
        isinstance(coordinates[0], str)
        and isinstance(coordinates[1], int)
        and isinstance(coordinates[2], int)
    ):
        raise ValueError(
            "Your coordinates must be a single tuple of types (str, int, int)."
        )
    if not (coordinates[1] < coordinates[2]):
        raise ValueError(
            f"End coordinate {coordinates[2]} should be bigger than start coordinate {coordinates[1]}"
        )

    # Get locations of the values given the binning
    if bin_size:
        positions = np.arange(
            start=coordinates[1] + bin_size / 2, stop=coordinates[2], step=bin_size
        )
    else:
        positions = np.arange(coordinates[1], coordinates[2])

    # Get values
    values = _extract_tracks_from_bigwig(
        bigwig_file, [coordinates], bin_size, target, missing, oob
    ).squeeze()

    return values, positions


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
