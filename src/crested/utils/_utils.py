"""General utility functions for the package."""

from __future__ import annotations

import os
import random
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger

from crested._genome import Genome, _resolve_genome
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

    Will return a numpy array with shape (1, len(sequence), 4) if expand_dim is True, otherwise (len(sequence),4).
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

    Can be passed as the 'enhancer_optimizer' argument to :func:`crested.tl.enhancer_design_in_silico_evolution`

    Parameters
    ----------
    optimize_func
        Function to optimize the mutated sequence based on the original prediction.

    See Also
    --------
    crested.tl.enhancer_design_in_silico_evolution
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


def _detect_input_type(input: str | list[str] | np.array | AnnData) -> str:
    """
    Detect the type of input provided.

    Parameters
    ----------
    input
        The input to detect the type of.

    Returns
    -------
    One of ['sequence', 'region', 'anndata', 'array'], indicating the input type.
    """
    dna_pattern = re.compile("^[ACGTNacgtn]+$")
    if isinstance(input, AnnData):
        return "anndata"
    elif isinstance(input, list):
        if all(":" in str(item) for item in input):  # List of regions
            return "region"
        elif all(
            isinstance(item, str) and dna_pattern.match(item) for item in input
        ):  # List of sequences
            return "sequence"
        else:
            raise ValueError(
                "List input must contain only valid region strings (chrom:var-end) or DNA sequences."
            )
    elif isinstance(input, str):
        if ":" in input:  # Single region
            return "region"
        elif dna_pattern.match(input):  # Single DNA sequence
            return "sequence"
        else:
            raise ValueError(
                "String input must be a valid region string (chrom:var-end) or DNA sequence."
            )
    elif isinstance(input, np.ndarray):
        if input.ndim == 3:
            return "array"
        else:
            raise ValueError("Input one hot array must have shape (N, L, 4).")
    else:
        raise ValueError(
            "Unsupported input type. Must be AnnData, str, list, or np.ndarray."
        )


def _transform_input(input, genome: Genome | os.PathLike | None = None) -> np.ndarray:
    """
    Transform the input into a one-hot encoded matrix based on its type.

    Parameters
    ----------
    input
        Input data to preprocess. Can be a sequence, list of sequences, region, list of regions, or an AnnData object.
    genome
        Genome or Path to the genome file. Required if no genome is registered and input is a region or AnnData.

    Returns
    -------
    One-hot encoded matrix of shape (N, L, 4), where N is the number of sequences/regions and L is the sequence length.
    """
    input_type = _detect_input_type(input)

    if input_type == "anndata":
        genome = _resolve_genome(genome)
        regions = list(input.var_names)
        sequences = [genome.fetch(region=region) for region in regions]
    elif input_type == "region":
        genome = _resolve_genome(genome)
        regions = input if isinstance(input, list) else [input]
        sequences = [genome.fetch(region=region) for region in regions]
    elif input_type == "sequence":
        sequences = input if isinstance(input, list) else [input]
    elif input_type == "array":
        assert input.ndim == 3, "Input one hot array must have shape (N, L, 4)."
        return input

    one_hot_data = np.array(
        [one_hot_encode_sequence(seq, expand_dim=False) for seq in sequences]
    )
    return one_hot_data


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
    regions: str | list[str],
    genome: os.PathLike | Genome | None = None,
    uppercase: bool = True,
) -> list[str]:
    """
    Fetch sequences from a genome file for a list of regions using pysam.

    Regions should be formatted as "chr:start-end".

    Parameters
    ----------
    regions
        List of regions to fetch sequences for.
    genome
        Path to the genome fasta or Genome instance or None.
        If None, will look for a registered genome object.
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
    genome = _resolve_genome(genome)
    seqs = []
    for region in regions:
        seq = genome.fetch(region=region)
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


def calculate_nucleotide_distribution(
    input: str | list[str] | np.ndarray | AnnData,
    genome: Genome | os.PathLike | None = None,
    per_position: bool = False,
    n_regions: int | None = None,
) -> np.ndarray:
    """
    Calculate the nucleotide distribution of a genome in a set of regions or sequences.

    Parameters
    ----------
    input
        Input data to calculate the ACGT distribution of. Can be a (list of) sequence(s), a (list of) region name(s), a matrix of one hot encodings (N, L, 4), or an AnnData object with region names as its var_names.
    genome
        The genome object or path to the genome fasta file. Required if input is a region or AnnData.
    per_position
        If True, calculate the nucleotide distribution per position in the sequence instead of over the whole sequence.
    n_regions
        Randomly sample n_regions from the input. If None, all inputs are used.
        This is useful for large datasets to speed up the calculation.

    Returns
    -------
    The nucleotide distribution as an array of floats (4,) in order A, C, G, T if per_position is False.
    Else, it returns an array of shape (L, 4) with the nucleotide distribution per position.
    """
    one_hots = _transform_input(input, genome)  # (N, L, 4)

    if n_regions is not None:
        random_sample = random.sample(range(one_hots.shape[0]), n_regions)
        one_hots = one_hots[random_sample]

    if per_position:
        distribution = np.mean(one_hots, axis=0).astype("float64")
    else:
        distribution = np.mean(one_hots, axis=(0, 1)).astype("float64")

    # get rid of floating point errors by normalizing
    return distribution / distribution.sum(axis=-1, keepdims=True)
