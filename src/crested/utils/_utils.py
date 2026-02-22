"""General utility functions for the package."""

from __future__ import annotations

import os
import random
import re

import numpy as np
from anndata import AnnData

from crested._genome import Genome, _resolve_genome
from crested._io import _extract_tracks_from_bigwig
from crested.utils._seq_utils import one_hot_encode_sequence


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


def _transform_input(
    input, genome: Genome | str | os.PathLike | None = None
) -> np.ndarray:
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
        sequences = [genome.fetch(region=region) for region in input.var_names]
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


def fetch_sequences(
    regions: str | list[str],
    genome: str | os.PathLike | Genome | None = None,
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
    bigwig_file: str | os.PathLike,
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
    >>> values, positions = crested.utils.read_bigwig_region(
    ...     bigwig_file="path/to/bigwig",
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
    genome: Genome | str | os.PathLike | None = None,
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


