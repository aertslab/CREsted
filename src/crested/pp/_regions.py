"""Region processing functions."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import pandas as pd
from anndata import AnnData
from loguru import logger

from crested import _conf as conf
from crested.utils._logging import log_and_raise


def _read_chromsizes(chromsizes_file: PathLike) -> dict[str, int]:
    """Read chromsizes file into a dictionary."""
    chromsizes = pd.read_csv(
        chromsizes_file, sep="\t", header=None, names=["chr", "size"]
    )
    chromsizes_dict = chromsizes.set_index("chr")["size"].to_dict()
    return chromsizes_dict


def change_regions_width(
    adata: AnnData,
    width: int,
    chromsizes_file: PathLike | None = None,
) -> None:
    """
    Change the widths of all regions in the adata object.

    The new region will have the same center as the original region.
    Modifies the adata.var and adata.var_names in place.
    This function is useful when you want to train on a wider/narrower region than the
    original consensus regions.

    Parameters
    ----------
    adata
        The AnnData object containing the regions as 'chr:start-end' to be modified in the .var_names.
    width
        The new width of the regions.
    chromsizes_file
        File path of the chromsizes file. Used for checking if the new regions are within the chromosome boundaries.

    Returns
    -------
    The AnnData object with the modified regions.

    Example
    -------
    >>> crested.pp.change_regions_width(adata, width=1000)
    """

    @log_and_raise(FileNotFoundError)
    def _check_input_params(chromsizes_file):
        if chromsizes_file is not None:
            chromsizes_file = Path(chromsizes_file)
            if not chromsizes_file.is_file():
                raise FileNotFoundError(f"File '{chromsizes_file}' not found")
        if chromsizes_file is None and not conf.genome:
            logger.warning(
                "Chromsizes file not provided. Will not check if regions are within chromosomes",
                stacklevel=1,
            )

    _check_input_params(chromsizes_file=chromsizes_file)

    if chromsizes_file is not None:
        chromsizes = _read_chromsizes(chromsizes_file)
    elif conf.genome:
        chromsizes = conf.genome.chrom_sizes
    else:
        chromsizes = None

    centers = (adata.var["start"] + adata.var["end"]) / 2
    half_width = width / 2

    adata.var = adata.var.copy()

    adata.var["start"] = (centers - half_width).astype(int)
    adata.var["end"] = (centers + half_width).astype(int)

    adata.var_names = adata.var.apply(
        lambda row: f"{row['chr']}:{row['start']}-{row['end']}", axis=1
    )

    # Check if regions are within the chromosome boundaries
    if chromsizes is not None:
        regions_to_keep = list(adata.var_names.copy())
        for idx, row in adata.var.iterrows():
            chr_name = row["chr"]
            start, end = row["start"], row["end"]
            if start < 0 or end > chromsizes.get(chr_name, float("inf")):
                logger.warning(
                    f"Region {idx} with coordinates {chr_name}:{start}-{end} is out of bounds for chromosome {chr_name}. Removing region."
                )
                regions_to_keep.remove(idx)
        if len(regions_to_keep) < len(adata.var_names):
            adata._inplace_subset_var(regions_to_keep)

    adata.var_names.name = "region"
