from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from crested._genome import Genome, _resolve_genome
from crested._io import _extract_tracks_from_bigwig


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
        seq = genome.fetch(region = region)
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


