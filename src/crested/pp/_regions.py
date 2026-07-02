"""Region processing functions."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import pandas as pd
from anndata import AnnData
from loguru import logger

from crested import Genome
from crested import _conf as conf
from crested.utils import parse_region
from crested.utils._logging import log_and_raise


def _read_chromsizes(chromsizes_file: str | PathLike) -> dict[str, int]:
    """Read chromsizes file into a dictionary."""
    chromsizes = pd.read_csv(
        chromsizes_file, sep="\t", header=None, names=["chr", "size"]
    )
    chromsizes_dict = chromsizes.set_index("chr")["size"].to_dict()
    return chromsizes_dict


def change_regions_width(
    adata: AnnData,
    width: int,
    chromsizes_file: str | PathLike | Genome | None = None,
    inplace: bool = True,
) -> AnnData | None:
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
        File path of the chromsizes file or Genome object. Used for checking if the new regions are within the chromosome boundaries.
        If not provided, uses chromsizes from the registered Genome object, and if that doesn't exist either, doesn't check against chromosome boundaries.
    inplace
        Perform computation and modify `adata` in-place or return a resulting copy of the `adata` instead.

    Returns
    -------
    If `inplace=True` (default), modifies the anndata in-place and doesn't return anything.
    If `inplace=False`, returns the AnnData object with the modified regions.

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

    # Handle pre-resizing checks
    _check_input_params(chromsizes_file=chromsizes_file)

    if chromsizes_file is not None:
        if isinstance(chromsizes_file, Genome):
            chromsizes = chromsizes_file.chrom_sizes
        else:
            chromsizes = _read_chromsizes(chromsizes_file)
    elif conf.genome:
        chromsizes = conf.genome.chrom_sizes
    else:
        chromsizes = None

    if adata.var_names[0].count(':') == 1:
        stranded = False
    elif adata.var_names[0].count(':') == 2:
        stranded = True
    else:
        raise ValueError("Region names must follow 'chr:start-end' or 'chr:start-end:strand' layout.")

    # Copy if doing inplace
    if not inplace:
        adata = adata.copy()
    adata.var = adata.var.copy()

    # Create new values and record spacing for all regions
    half_width = width / 2
    new_starts, new_ends, new_names = [], [], []
    regions_to_keep = []
    for region_name in adata.var_names:
        # Resize regions
        chrom, start, end, strand = parse_region(region_name)
        center = (start + end)/2
        new_start, new_end = int(center-half_width), int(center+half_width)
        new_name = f"{chrom}:{int(center-half_width)}-{int(center+half_width)}"
        if stranded:
            new_name += f":{strand}"
        new_starts.append(new_start)
        new_ends.append(new_end)
        new_names.append(new_name)

        # Check chromosome boundaries on the resized coordinates (not the
        # originals): a region whose centered/widened window runs off a contig
        # edge must be dropped even if the original peak was in-bounds.
        if chromsizes is not None:
            if new_start < 0 or new_end > chromsizes.get(chrom, float("inf")):
                logger.warning(
                    f"Region {region_name} with new coordinates {chrom}:{new_start}-{new_end} is out of bounds for chromosome {chrom}. Removing region."
                )
            else:
                regions_to_keep.append(new_name)

    # Set new values in adata
    adata.var['unresized_index'] = adata.var_names
    adata.var.index = new_names
    adata.var['start'] = new_starts
    adata.var['end'] = new_ends
    adata.var_names.name = "region"

    # Filter out oversized regions
    if chromsizes is not None and (len(regions_to_keep) < len(adata.var_names)):
        if inplace:
            adata._inplace_subset_var(regions_to_keep)
        else:
            adata = adata[:, regions_to_keep].copy()


    if not inplace:
        return adata
