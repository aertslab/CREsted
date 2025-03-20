"""Module for splitting datasets into train, validation, and test sets."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger

__all__ = ["train_val_test_split"]


def _split_by_chromosome_auto(
    regions: list[str],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    random_state: int | None = None,
) -> pd.Series:
    """Split the dataset based on chromosome, automatically selecting chromosomes for val and test sets.

    Parameters
    ----------
        regions (list): List of region strings formatted as 'chr:start-end'.
        val_fraction (float): Fraction of regions to include in val set.
        test_fraction (float): Fraction of regions to include in test set.

    Returns
    -------
        pd.Series: Series with the split assignment for each region.
    """
    chrom_count = defaultdict(int)
    for region in regions:
        if ":" not in region:
            raise ValueError(
                f"Region names should start with the chromosome name, bound by a colon (:). Offending region: {region}"
            )
        chrom = region.split(":")[0]
        chrom_count[chrom] += 1

    total_regions = sum(chrom_count.values())
    target_val_size = int(val_fraction * total_regions)
    target_test_size = int(test_fraction * total_regions)

    chromosomes = list(chrom_count.keys())
    np.random.seed(seed=random_state)
    np.random.shuffle(chromosomes)

    val_chroms = set()
    test_chroms = set()
    current_val_size = 0
    current_test_size = 0

    for chrom in chromosomes:
        if current_val_size < target_val_size:
            val_chroms.add(chrom)
            current_val_size += chrom_count[chrom]
        elif current_test_size < target_test_size:
            test_chroms.add(chrom)
            current_test_size += chrom_count[chrom]
        if (
            current_val_size >= target_val_size
            and current_test_size >= target_test_size
        ):
            break

    indices = {"train": [], "val": [], "test": []}
    for i, region in enumerate(regions):
        chrom = region.split(":")[0]
        if chrom in val_chroms:
            indices["val"].append(i)
        elif chrom in test_chroms:
            indices["test"].append(i)
        else:
            indices["train"].append(i)

    split = pd.Series("train", index=regions)
    split.iloc[indices["val"]] = "val"
    split.iloc[indices["test"]] = "test"
    return split


def _split_by_chromosome(
    regions: list[str], val_chroms: list[str], test_chroms: list[str]
) -> pd.Series:
    """
    Split the dataset based on selected chromosomes.

    If the same chromosomes are supplied
    to both val and test, then the regions will be divided evenly between.

    Parameters
    ----------
        regions (list): List of region strings formatted as 'chr:start-end'.
        val_chroms (list): List of chromosome names to include in val set.
        test_chroms (list): List of chromosome names to include in test set.

    Returns
    -------
        pd.Series: Series with the split assignment for each region.
    """
    indices = {"train": [], "val": [], "test": []}
    all_chroms = {region.split(":")[0] for region in regions}

    if not set(val_chroms).issubset(all_chroms):
        raise ValueError("One or more val chromosomes not found in regions.")
    if not set(test_chroms).issubset(all_chroms):
        raise ValueError("One or more test chromosomes not found in regions.")

    # Split
    overlap_chroms = set(val_chroms) & set(test_chroms)
    val_chroms = set(val_chroms) - overlap_chroms
    test_chroms = set(test_chroms) - overlap_chroms
    chrom_counter = {}

    for i, region in enumerate(regions):
        chrom = region.split(":")[0]
        if chrom in overlap_chroms:
            if chrom not in chrom_counter:
                chrom_counter[chrom] = 0
            if chrom_counter[chrom] % 2 == 0:
                indices["val"].append(i)
            else:
                indices["test"].append(i)
            chrom_counter[chrom] += 1
        elif chrom in val_chroms:
            indices["val"].append(i)
        elif chrom in test_chroms:
            indices["test"].append(i)
        else:
            indices["train"].append(i)

    split = pd.Series("train", index=regions)
    split.iloc[indices["val"]] = "val"
    split.iloc[indices["test"]] = "test"
    return split


def _split_by_regions(
    regions: list[str],
    val_size: float = 0.1,
    test_size: float = 0.1,
    shuffle: bool = True,
    random_state: None | int = None,
) -> pd.Series[str]:
    """Split regions into train, val, and test sets based on region names.

    Parameters
    ----------
    regions : list
        List of region names.
    val_size : float
        Proportion of the dataset to include in the validation split.
    test_size : float
        Proportion of the dataset to include in the test split.
    shuffle : bool
        Whether or not to shuffle the data before splitting.
    random_state : int
        Random seed for shuffling.

    Returns
    -------
        pd.Series: Series with the split assignment for each region.
    """
    n_samples = len(regions)

    if shuffle:
        np.random.seed(seed=random_state)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    test_n = math.ceil(n_samples * test_size)
    val_n = math.ceil(n_samples * val_size)

    split = pd.Series("train", index=range(n_samples))
    split.iloc[indices[:test_n]] = "test"
    split.iloc[indices[test_n : test_n + val_n]] = "val"
    split.iloc[indices[test_n + val_n :]] = "train"

    split = pd.Series("train", index=regions)
    split.iloc[indices[:test_n]] = "test"
    split.iloc[indices[test_n : test_n + val_n]] = "val"

    return split


def train_val_test_split(
    adata: AnnData,
    strategy: str = "region",
    val_size: float = 0.1,
    test_size: float = 0.1,
    val_chroms: list[str] = None,
    test_chroms: list[str] = None,
    shuffle: bool = True,
    random_state: None | int = None,
) -> None:
    """
    Add 'train/val/test' split column to AnnData object.

    Adds a new column `split` to the `.var` DataFrame of the AnnData object,
    indicating whether each sample should be part of the training, validation, or test set
    based on the chosen splitting strategy.

    Note
    ----
    Model training always requires a `split` column in the `.var` DataFrame.

    Parameters
    ----------
    adata
        AnnData object to which the 'train/val/test' split column will be added.
    strategy
        strategy of split. Either 'region', 'chr' or 'chr_auto'. If 'chr' or 'chr_auto', the anndata's var_names should
        contain the chromosome name at the start, followed by a `:` (e.g. I:2000-2500 or chr3:10-20:+).

        region: Split randomly on region indices.

        chr: Split based on provided chromosomes.

        chr_auto: Automatically select chromosomes for val and test sets based on val and test size.

        If strategy 'chr', it's also possible to provide the same chromosome(s) to both val_chroms and test_chroms.
        In this case, the regions will be divided evenly between the two sets.
    val_size
        Proportion of the training dataset to include in the validation split.
    test_size
        Proportion of the dataset to include in the test split.
    val_chroms
        List of chromosomes to include in the validation set. Required if strategy='chr'.
    test_chroms
        List of chromosomes to include in the test set. Required if strategy='chr'.
    shuffle
        Whether or not to shuffle the data before splitting (when strategy='region').
    random_state
        Random_state affects the ordering of the indices when shuffling in regions or
        auto splitting on chromosomes.

    Returns
    -------
    Adds a new column inplace to `adata.var`:
        'split': 'train', 'val', or 'test'

    Examples
    --------
    >>> crested.train_val_test_split(
    ...     adata,
    ...     strategy="region",
    ...     val_size=0.1,
    ...     test_size=0.1,
    ...     shuffle=True,
    ...     random_state=42,
    ... )

    >>> crested.train_val_test_split(
    ...     adata,
    ...     strategy="chr",
    ...     val_chroms=["chr1", "chr2"],
    ...     test_chroms=["chr3", "chr4"],
    ... )
    """
    # Input checks
    if strategy not in ["region", "chr", "chr_auto"]:
        raise ValueError("`strategy` should be either 'region','chr', or 'chr_auto'")
    if strategy in ["region", "chr_auto"] and not 0 <= val_size <= 1:
        raise ValueError("`val_size` should be a float between 0 and 1.")
    if strategy in ["region", "chr_auto"] and not 0 <= test_size <= 1:
        raise ValueError("`test_size` should be a float between 0 and 1.")
    if (strategy == "region") and (val_chroms is not None or test_chroms is not None):
        logger.warning(
            "`val_chroms` and `test_chroms` provided but splitting strategy is 'region'. Will use 'chr' strategy instead."
        )
        strategy = "chr"
    if strategy == "chr":
        if val_chroms is None or test_chroms is None:
            raise ValueError(
                "If `strategy` is 'chr', `val_chroms` and `test_chroms` should be provided."
            )

    # Split
    regions = list(adata.var_names)

    if strategy == "region":
        split = _split_by_regions(regions, val_size, test_size, shuffle, random_state)
    elif strategy == "chr":
        split = _split_by_chromosome(
            regions, val_chroms=val_chroms, test_chroms=test_chroms
        )
    elif strategy == "chr_auto":
        split = _split_by_chromosome_auto(regions, val_size, test_size, random_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata.var["split"] = split
