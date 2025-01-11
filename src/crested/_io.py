"""I/O functions for importing beds and bigWigs into AnnData objects."""

from __future__ import annotations

import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from os import PathLike
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pybigtools
from anndata import AnnData
from loguru import logger
from scipy.sparse import csr_matrix
from tqdm import tqdm
from crested import _conf as conf


def _sort_files(filename: PathLike):
    """Sorts files.

    Prioritizes numeric extraction from filenames of the format 'Class_X.bed' (X=int).
    Other filenames are sorted alphabetically, with 'Class_' files coming last if numeric extraction fails.
    """
    filename = Path(filename)
    parts = filename.stem.split("_")

    if len(parts) > 1:
        try:
            return (False, int(parts[1]))
        except ValueError:
            # If the numeric part is not an integer, handle gracefully
            return (True, filename.stem)

    # Return True for the first element to sort non-'Class_X' filenames alphabetically after 'Class_X'
    return (
        True,
        filename.stem,
    )


def _custom_region_sort(region: str) -> tuple[int, int, int]:
    """Sort regions in the format chr:start-end."""
    chrom, pos = region.split(":")
    start, _ = map(int, pos.split("-"))

    # check if the chromosome part contains digits
    numeric_match = re.match(r"chr(\d+)|chrom(\d+)", chrom, re.IGNORECASE)

    if numeric_match:
        chrom_num = int(numeric_match.group(1) or numeric_match.group(2))
        return (0, chrom_num, start)
    else:
        return (1, chrom, start)


def _read_chromsizes(chromsizes_file: PathLike) -> dict[str, int]:
    """Read chromsizes file into a dictionary."""
    chromsizes = pd.read_csv(
        chromsizes_file, sep="\t", header=None, names=["chr", "size"]
    )
    chromsizes_dict = chromsizes.set_index("chr")["size"].to_dict()
    return chromsizes_dict


def _extract_values_from_bigwig(
    bw_file: PathLike, bed_file: PathLike, target: str
) -> np.ndarray:
    """Extract target values from a bigWig file for regions specified in a BED file."""
    if isinstance(bed_file, Path):
        bed_file = str(bed_file)
    if isinstance(bw_file, Path):
        bw_file = str(bw_file)

    # Get chromosomes available in bigWig file.
    with pybigtools.open(bw_file, "r") as bw:
        chromosomes_in_bigwig = set(bw.chroms())

    # Create temporary BED file with only BED entries that are in the bigWig file.
    temp_bed_file = tempfile.NamedTemporaryFile()
    bed_entries_to_keep_idx = []

    with open(bed_file) as fh:
        for idx, line in enumerate(fh):
            chrom = line.split("\t", 1)[0]
            if chrom in chromosomes_in_bigwig:
                temp_bed_file.file.write(line.encode("utf-8"))
                bed_entries_to_keep_idx.append(idx)
        temp_bed_file.file.flush()

    total_bed_entries = idx + 1
    bed_entries_to_keep_idx = np.array(bed_entries_to_keep_idx, np.intp)

    # Branch logic for target
    if target == "mean":
        with pybigtools.open(bw_file, "r") as bw:
            values = np.fromiter(
                bw.average_over_bed(bed=temp_bed_file.name, names=None, stats="mean0"),
                dtype=np.float32,
            )
    elif target == "max":
        with pybigtools.open(bw_file, "r") as bw:
            values = np.fromiter(
                bw.average_over_bed(bed=temp_bed_file.name, names=None, stats="max"),
                dtype=np.float32,
            )
    elif target == "count":
        with pybigtools.open(bw_file, "r") as bw:
            values = np.fromiter(
                bw.average_over_bed(bed=temp_bed_file.name, names=None, stats="sum"),
                dtype=np.float32,
            )
    elif target == "logcount":
        with pybigtools.open(bw_file, "r") as bw:
            values = np.log1p(
                np.fromiter(
                    bw.average_over_bed(
                        bed=temp_bed_file.name, names=None, stats="sum"
                    ),
                    dtype=np.float32,
                )
            )

    else:
        raise ValueError(f"Unsupported target '{target}'")

    temp_bed_file.close()

    # Now handle missing chromosome lines
    if target == "raw":
        # 'values' is 2D [n_valid_regions, region_length or max_length].
        # We have total_bed_entries lines in original bed, but only n_valid in 'values'.
        # If we want to keep shape [n_regions, region_length] for the final result,
        # we must build a bigger 2D array with shape [total_bed_entries, max_length].
        all_data = np.full((total_bed_entries, values.shape[1]), np.nan, dtype=np.float32)
        all_data[bed_entries_to_keep_idx, :] = values
        return all_data

    else:
        # 'values' is 1D
        if values.shape[0] != total_bed_entries:
            # Set all values for BED entries for which the chromosome was not in bigWig to NaN.
            all_values = np.full(total_bed_entries, np.nan, dtype=np.float32)
            all_values[bed_entries_to_keep_idx] = values
            return all_values
        else:
            return values

def _extract_tracks_from_bigwig(
    bw_file: PathLike,
    coordinates: list[tuple[str, int, int]],
    bin_size: int | None = None,
    target: str = "mean",
    missing: float = 0.0,
    oob: float = 0.0,
    exact: bool = True,
) -> np.ndarray:
    """
    Extract per-base or binned pair values of a list of genomic ranges from a bigWig file.

    Expects all coordinate pairs to be the same length.

    bigwig_file
        Path to the bigWig file.
    coordinates
        A list of tuples looking like (chr, start, end).
    bin_size
        If set, the returned values are mean-binned at this resolution.
    target
        How to summarize the values per bin, when binning. Can be 'mean', 'min', or 'max'.
    missing
        Fill-in value for unreported data in valid regions. Default is 0.
    oob
        Fill-in value for out-of-bounds regions.
    exact
        Whether to always return the exact values, or to use the built-in zoom levels to interpolate, when binning.
        Setting exact = False leads to a slight speed advantage, but slight loss in accuracy.

    Returns a numpy array of values from the bigWig file of shape [n_coordinates, n_base_pairs] or [n_coordinates, n_base_pairs//bin_size] if bin_size is set.
    """
    # Wrapper around pybigtools.BBIRead.values().

    # Check that all are same size by iterating and checking with predecessor
    prev_region_length = coordinates[0][2] - coordinates[0][1]
    for region in coordinates:
        region_length = region[2] - region[1]
        if region_length != prev_region_length:
            raise ValueError(
                f"All coordinate pairs should be the same length. Coordinate pair {region[0]}:{region[1]}-{region[2]} is not {prev_region_length}bp, but {region_length}bp."
            )
        prev_region_length = region_length

    # Check that length is divisible by bin size
    if bin_size and (region_length % bin_size != 0):
        raise ValueError(
            f"All region lengths must be divisible by bin_size. Region length {region_length} is not divisible by bin size {bin_size}."
        )

    # Calculate length (for array creation) and bins (for argument to bw.values)
    binned_length = region_length // bin_size if bin_size else region_length
    bins = region_length // bin_size if bin_size else None

    # Open the bigWig file
    with pybigtools.open(bw_file, "r") as bw:
        results = []
        for region in coordinates:
            arr = np.empty(
                binned_length, dtype="float64"
            )  # pybigtools returns values in float64
            chrom, start, end = region

            # Extract values
            results.append(
                bw.values(
                    chrom,
                    start,
                    end,
                    bins=bins,
                    summary=target,
                    exact=exact,
                    missing=missing,
                    oob=oob,
                    arr=arr,
                )
            )

    return np.vstack(results)


def _read_consensus_regions(
    regions_file: PathLike, chromsizes_dict: dict | None = None
) -> pd.DataFrame:
    """Read consensus regions BED file and filter out regions not within chromosomes."""
    if chromsizes_dict is None and not conf.genome:
        logger.warning(
            "Chromsizes file not provided. Will not check if regions are within chromosomes",
            stacklevel=1,
        )
    consensus_peaks = pd.read_csv(
        regions_file,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        dtype={0: str, 1: "Int32", 2: "Int32"},
    )
    consensus_peaks.columns = ["chr","start","end"]
    consensus_peaks["region"] = (
        consensus_peaks["chr"].astype(str)
        + ":"
        + consensus_peaks["start"].astype(str)
        + "-"
        + consensus_peaks["end"].astype(str)
    )
    if chromsizes_dict:
        pass
    elif conf.genome:
        chromsizes_dict = conf.genome.chrom_sizes
    else:
        return consensus_peaks

    valid_mask = consensus_peaks.apply(
        lambda row: row["chr"] in chromsizes_dict
        and row["start"] >= 0
        and row["end"] <= chromsizes_dict[row[0]],
        axis=1,
    )
    consensus_peaks_filtered = consensus_peaks[valid_mask]

    if len(consensus_peaks) != len(consensus_peaks_filtered):
        logger.warning(
            f"Filtered {len(consensus_peaks) - len(consensus_peaks_filtered)} consensus regions (not within chromosomes)",
        )
    return consensus_peaks_filtered


def _create_temp_bed_file(
    consensus_peaks: pd.DataFrame, target_region_width: int, adjust = True
) -> str:
    """Adjust consensus regions to a target width and create a temporary BED file."""
    adjusted_peaks = consensus_peaks.copy()
    if adjust:
        adjusted_peaks[1] = adjusted_peaks.apply(
            lambda row: max(0, row[1] - (target_region_width - (row[2] - row[1])) // 2),
            axis=1,
        )
        adjusted_peaks[2] = adjusted_peaks[1] + target_region_width
        adjusted_peaks[1] = adjusted_peaks[1].astype(int)
        adjusted_peaks[2] = adjusted_peaks[2].astype(int)

    # Create a temporary BED file
    temp_bed_file = "temp_adjusted_regions.bed"
    adjusted_peaks.to_csv(temp_bed_file, sep="\t", header=False, index=False)
    return temp_bed_file


def _check_bed_file_format(bed_file: PathLike) -> None:
    """Check if the BED file is in the correct format."""
    with open(bed_file) as f:
        first_line = f.readline().strip()
    # check if at least three columns are found
    if len(first_line.split("\t")) < 3:
        raise ValueError(
            f"BED file '{bed_file}' is not in the correct format. "
            "Expected at least three tab-seperated columns."
        )
    pattern = r".*\t\d+\t\d+.*"
    if not re.match(pattern, first_line):
        raise ValueError(
            f"BED file '{bed_file}' is not in the correct format. "
            "Expected columns 2 and 3 to contain integers."
        )


def import_beds(
    beds_folder: PathLike,
    regions_file: PathLike | None = None,
    chromsizes_dict: dict | None = None,
    classes_subset: list | None = None,
    remove_empty_regions: bool = True,
    compress: bool = False,
) -> AnnData:
    """
    Import beds and optionally consensus regions BED files into AnnData format.

    Expects the folder with BED files where each file is named {class_name}.bed
    The result is an AnnData object with classes as rows and the regions as columns,
    with the .X values indicating whether a region is open in a class.

    Note
    ----
    This is the default function to import topic BED files coming from running pycisTopic
    (https://pycistopic.readthedocs.io/en/latest/) on your data.
    The result is an AnnData object with topics as rows and consensus region as columns,
    with binary values indicating whether a region is present in a topic.

    Parameters
    ----------
    beds_folder
        Folder path containing the BED files.
    regions_file
        File path of the consensus regions BED file to use as columns in the AnnData object.
        If None, the regions will be extracted from the files.
    classes_subset
        List of classes to include in the AnnData object. If None, all files
        will be included.
        Classes should be named after the file name without the extension.
    chromsizes_dict
        dict of chromsizes.  Used for checking if the new regions are within the chromosome boundaries.
        If not provided, will look for a registered genome object.
    remove_empty_regions
        Remove regions that are not open in any class (only possible if regions_file is provided)
    compress
        Compress the AnnData.X matrix. If True, the matrix will be stored as
        a sparse matrix. If False, the matrix will be stored as a dense matrix.

        WARNING: Compressing the matrix currently makes training very slow and is never recommended.
        We're still investigating a way around.

    Returns
    -------
    AnnData object with classes as rows and peaks as columns.

    Example
    -------
    >>> anndata = crested.import_beds(
    ...     beds_folder="path/to/beds/folder/",
    ...     regions_file="path/to/regions.bed",
    ...     classes_subset=["Topic_1", "Topic_2"],
    ... )
    """
    beds_folder = Path(beds_folder)
    regions_file = Path(regions_file) if regions_file else None

    # Input checks
    if not beds_folder.is_dir():
        raise FileNotFoundError(f"Directory '{beds_folder}' not found")
    if (regions_file is not None) and (not regions_file.is_file()):
        raise FileNotFoundError(f"File '{regions_file}' not found")
    if classes_subset is not None:
        for classname in classes_subset:
            if not any(beds_folder.glob(f"{classname}.bed")):
                raise FileNotFoundError(f"'{classname}' not found in '{beds_folder}'")

    if regions_file:
        # Read consensus regions BED file and filter out regions not within chromosomes
        _check_bed_file_format(regions_file)
        consensus_peaks = _read_consensus_regions(regions_file, chromsizes_dict)

        binary_matrix = pd.DataFrame(0, index=[], columns=consensus_peaks["region"])
        file_paths = []

        # Which regions are present in the consensus regions
        logger.info(
            f"Reading bed files from {beds_folder} and using {regions_file} as var_names..."
        )
        for file in sorted(beds_folder.glob("*.bed"), key=_sort_files):
            class_name = file.stem
            if classes_subset is None or class_name in classes_subset:
                class_regions = pd.read_csv(
                    file, sep="\t", header=None, usecols=[0, 1, 2]
                )
                class_regions["region"] = (
                    class_regions[0].astype(str)
                    + ":"
                    + class_regions[1].astype(str)
                    + "-"
                    + class_regions[2].astype(str)
                )

                # Create binary row for the current topic
                binary_row = binary_matrix.columns.isin(class_regions["region"]).astype(
                    int
                )
                binary_matrix.loc[class_name] = binary_row
                file_paths.append(str(file))

    # else, get regions from the bed files
    else:
        file_paths = []
        all_regions = set()

        # Collect all regions from the BED files
        logger.info(
            f"Reading bed files from {beds_folder} without consensus regions..."
        )
        for file in sorted(beds_folder.glob("*.bed"), key=_sort_files):
            class_name = file.stem
            if classes_subset is None or class_name in classes_subset:
                _check_bed_file_format(file)
                class_regions = pd.read_csv(
                    file, sep="\t", header=None, usecols=[0, 1, 2]
                )
                class_regions["region"] = (
                    class_regions[0].astype(str)
                    + ":"
                    + class_regions[1].astype(str)
                    + "-"
                    + class_regions[2].astype(str)
                )
                all_regions.update(class_regions["region"].tolist())
                file_paths.append(str(file))

        # Convert set to sorted list
        all_regions = sorted(all_regions, key=_custom_region_sort)
        binary_matrix = pd.DataFrame(0, index=[], columns=all_regions)

        # Populate the binary matrix
        for file in file_paths:
            class_name = Path(file).stem
            class_regions = pd.read_csv(file, sep="\t", header=None, usecols=[0, 1, 2])
            class_regions["region"] = (
                class_regions[0].astype(str)
                + ":"
                + class_regions[1].astype(str)
                + "-"
                + class_regions[2].astype(str)
            )
            binary_row = binary_matrix.columns.isin(class_regions["region"]).astype(int)
            binary_matrix.loc[class_name] = binary_row

    ann_data = AnnData(
        binary_matrix,
    )

    ann_data.obs["file_path"] = file_paths
    ann_data.obs["n_open_regions"] = ann_data.X.sum(axis=1)
    ann_data.var["n_classes"] = ann_data.X.sum(axis=0)
    ann_data.var["chr"] = ann_data.var.index.str.split(":").str[0]
    ann_data.var["start"] = (
        ann_data.var.index.str.split(":").str[1].str.split("-").str[0]
    ).astype(int)
    ann_data.var["end"] = (
        ann_data.var.index.str.split(":").str[1].str.split("-").str[1]
    ).astype(int)

    if compress:
        ann_data.X = csr_matrix(ann_data.X)

    # Output checks
    classes_no_open_regions = ann_data.obs[ann_data.obs["n_open_regions"] == 0]
    if not classes_no_open_regions.empty:
        raise ValueError(
            f"{classes_no_open_regions.index} have 0 open regions in the consensus peaks"
        )
    regions_no_classes = ann_data.var[ann_data.var["n_classes"] == 0]
    if not regions_no_classes.empty:
        if remove_empty_regions:
            logger.warning(
                f"{len(regions_no_classes.index)} consensus regions are not open in any class. Removing them from the AnnData object. Disable this behavior by setting 'remove_empty_regions=False'",
            )
            ann_data = ann_data[:, ann_data.var["n_classes"] > 0]

    return ann_data


def import_bigwigs(
    bigwigs_folder: PathLike,
    regions_file: PathLike,
    chromsizes_dict: dict | None = None,
    target: str = "mean",
    target_region_width: int | None = None,
    compress: bool = False,
) -> AnnData:
    """
    Import bigWig files and consensus regions BED file into AnnData format.

    This format is required to be able to train a peak prediction model.
    The bigWig files target values are calculated for each region and and imported into an AnnData object,
    with the bigWig file names as .obs and the consensus regions as .var.
    Optionally, the target region width can be specified to extract values from a wider/narrower region around the consensus region,
    where the original region will still be used as the index.
    This is often useful to extract sequence information around the actual peak region.

    Parameters
    ----------
    bigwigs_folder
        Folder name containing the bigWig files.
    regions_file
        File name of the consensus regions BED file.
    chromsizes_dict
        Dictionary of chrom sizes. Used for checking if the new regions are within the chromosome boundaries.
        If not provided, will look for a registered genome object.
    target
        Target value to extract from bigwigs. Can be 'raw', 'mean', 'max', 'count', or 'logcount'
    target_region_width
        Width of region that the bigWig target value will be extracted from. If None, the
        consensus region width will be used.
    compress
        Compress the AnnData.X matrix. If True, the matrix will be stored as
        a sparse matrix. If False, the matrix will be stored as a dense matrix.

    Returns
    -------
    AnnData object with bigWigs as rows and peaks as columns.

    Example
    -------
    >>> anndata = crested.import_bigwigs(
    ...     bigwigs_folder="path/to/bigwigs",
    ...     regions_file="path/to/peaks.bed",
    ...     chromsizes_file="path/to/chrom.sizes",
    ...     target="max",
    ...     target_region_width=500,
    ... )
    """
    bigwigs_folder = Path(bigwigs_folder)
    regions_file = Path(regions_file)

    # Input checks
    if not bigwigs_folder.is_dir():
        raise FileNotFoundError(f"Directory '{bigwigs_folder}' not found")
    if not regions_file.is_file():
        raise FileNotFoundError(f"File '{regions_file}' not found")

    # Read consensus regions BED file and filter out regions not within chromosomes
    _check_bed_file_format(regions_file)
    consensus_peaks = _read_consensus_regions(regions_file, chromsizes_file)
    
    if target_region_width is not None:
        bed_file = _create_temp_bed_file(consensus_peaks, target_region_width)
    else:
        bed_file = regions_file

    bw_files = []
    for file in os.listdir(bigwigs_folder):
        file_path = os.path.join(bigwigs_folder, file)
        try:
            # Validate using pyBigTools (add specific validation if available)
            bw = pybigtools.open(file_path, "r")
            bw_files.append(file_path)
            bw.close()
        except ValueError:
            pass
        except pybigtools.BBIReadError:
            pass

    bw_files = sorted(bw_files)
    if not bw_files:
        raise FileNotFoundError(f"No valid bigWig files found in '{bigwigs_folder}'")

    # Process bigWig files in parallel and collect the results
    logger.info(f"Extracting values from {len(bw_files)} bigWig files...")
    all_results = []
    # with ProcessPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(
    #             _extract_values_from_bigwig,
    #             bw_file,
    #             bed_file,
    #             target,
    #         )
    #         for bw_file in bw_files
    #     ]
    #     for future in futures:
    #         all_results.append(future.result())

    for bw_file in bw_files:
        result = _extract_values_from_bigwig(bw_file, bed_file, target=target)
        all_results.append(result)

    if target_region_width is not None:
        os.remove(bed_file)

    data_matrix = np.vstack(all_results)

    # Prepare obs and var for AnnData
    obs_df = pd.DataFrame(
        data={"file_path": bw_files},
        index=[
            os.path.basename(file).rpartition(".")[0].replace(".", "_")
            for file in bw_files
        ],
    )
    var_df = pd.DataFrame(
        {
            "region": consensus_peaks["region"],
            "chr": consensus_peaks["region"].str.split(":").str[0],
            "start": (
                consensus_peaks["region"].str.split(":").str[1].str.split("-").str[0]
            ).astype(int),
            "end": (
                consensus_peaks["region"].str.split(":").str[1].str.split("-").str[1]
            ).astype(int),
        }
    ).set_index("region")

    # Create AnnData object
    adata = ad.AnnData(data_matrix, obs=obs_df, var=var_df)

    if compress:
        adata.X = csr_matrix(adata.X)

    # Output checks
    regions_no_values = adata.var[adata.X.sum(axis=0) == 0]
    if not regions_no_values.empty:
        logger.warning(
            f"{len(regions_no_values.index)} consensus regions have no values in any bigWig file",
        )

    return adata


import scipy
from scipy.sparse import csr_matrix
import anndata
from anndata._io.h5ad import read_elem, read_dataframe
from pathlib import Path
from typing import Literal
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Literal
from anndata._io.h5ad import *
from anndata import AnnData
import numpy as np
import h5py
import pybigtools
import numpy as np
from pathlib import Path
from os import PathLike
from crested._genome import Genome


class H5Source:
    """
    A lightweight reference to a dataset in an HDF5 file on disk.
    It does NOT hold data in memory, only the filename/path.
    """
    def __init__(self, filename: str, dataset_path: str):
        self.filename = filename
        self.dataset_path = dataset_path

    @property
    def shape(self):
        with h5py.File(self.filename, "r") as f:
            return f[self.dataset_path].shape

    def __getitem__(self, idx):
        with h5py.File(self.filename, "r") as f:
            return f[self.dataset_path][idx]


class LazyTensor(np.ndarray):
    """
    A 2D "indexable" view of a 3D+ dataset [rows, cols, coverage_length,...].
    For each (row, col), we retrieve a 1D coverage array of shape [coverage_length].
    """

    def __new__(
        cls,
        source,
        row_labels=None,
        col_labels=None,
    ):
        """
        source: An H5Source or similar, expected shape = (n_rows, n_cols, coverage_length).
        row_labels: optional list of row (track) labels
        col_labels: optional list of column (region) labels
        """
        obj = super().__new__(cls, shape=(), dtype=float)
        obj.source = source  # e.g. H5Source
        # We define the 2D "index shape" as (source.shape[0], source.shape[1])
        # coverage is the 3rd dimension
        obj._lazy_shape = (obj.source.shape[0], obj.source.shape[1])  

        # Create label -> index maps if provided
        if row_labels:
            obj.row_labels = {label: i for i, label in enumerate(row_labels)}
        else:
            obj.row_labels = None

        if col_labels:
            obj.col_labels = {label: i for i, label in enumerate(col_labels)}
        else:
            obj.col_labels = None

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.source = getattr(obj, "source", None)
        self._lazy_shape = getattr(obj, "_lazy_shape", None)
        self.row_labels = getattr(obj, "row_labels", None)
        self.col_labels = getattr(obj, "col_labels", None)

    def _get_indices(self, labels, label_map):
        """
        Convert label-based indices into integer indices.
        Returns array of float so we can store np.nan for missing labels.
        """
        if label_map is None:
            raise IndexError("Label-based indexing is not supported on this axis.")
        indices = []
        for label in labels:
            if label in label_map:
                indices.append(label_map[label])
            else:
                indices.append(np.nan)  # missing
        return np.array(indices, dtype=float)

    def __getitem__(self, index):
        """
        Expects 2D indexing, e.g. lazy_tensor[row_idx, col_idx],
        returning a coverage array of shape (len(row_idx), len(col_idx), coverage_length).
        """
        if not isinstance(index, tuple) or len(index) != 2:
            raise IndexError("LazyTensor expects 2D indexing, e.g. lazy_tensor[i, j].")

        row_index, col_index = index

        # 1) parse row_index
        if isinstance(row_index, str):
            row_index = self._get_indices([row_index], self.row_labels)
        elif isinstance(row_index, (list, np.ndarray)) and len(row_index) > 0 and isinstance(row_index[0], str):
            row_index = self._get_indices(row_index, self.row_labels)
        elif isinstance(row_index, slice):
            row_index = np.arange(*row_index.indices(self._lazy_shape[0]), dtype=float)
        elif isinstance(row_index, (int, np.integer)):
            row_index = np.array([row_index], dtype=float)
        else:
            row_index = np.array(row_index, dtype=float)

        # 2) parse col_index
        if isinstance(col_index, str):
            col_index = self._get_indices([col_index], self.col_labels)
        elif isinstance(col_index, (list, np.ndarray)) and len(col_index) > 0 and isinstance(col_index[0], str):
            col_index = self._get_indices(col_index, self.col_labels)
        elif isinstance(col_index, slice):
            col_index = np.arange(*col_index.indices(self._lazy_shape[1]), dtype=float)
        elif isinstance(col_index, (int, np.integer)):
            col_index = np.array([col_index], dtype=float)
        else:
            col_index = np.array(col_index, dtype=float)

        # 3) Build an output array => (n_rows, n_cols, coverage_length)
        coverage_length = self.source.shape[2]  # the 3rd dimension
        out_shape = (len(row_index), len(col_index), coverage_length)

        result = np.empty(out_shape, dtype=np.float32)
        result.fill(np.nan)

        # 4) read from source for each (r, c)
        for i, r in enumerate(row_index):
            for j, c in enumerate(col_index):
                if not np.isnan(r) and not np.isnan(c):
                    r_int = int(r)
                    c_int = int(c)
                    # read coverage from [r_int, c_int, :]
                    data_1d = self.source[r_int, c_int]
                    # data_1d is shape = (coverage_length,)
                    result[i, j, :] = data_1d

        return result

    @property
    def shape(self):
        # The 2D "index shape" is (n_tracks, n_regions)
        return self._lazy_shape

    def __repr__(self):
        return f"<LazyTensor shape={self._lazy_shape} + coverage>"

class LazyMatrix(np.ndarray):
    """
    A 2D NumPy array subclass. Indexing behaves like a normal matrix:
      - [i, j] => a scalar
      - [i, :] => a 1D array
      - [i:j, p:q] => a 2D array
    The actual data is fetched on demand from `source`.
    """

    def __new__(cls, source, dtype=float):
        obj = super().__new__(cls, shape=(), dtype=dtype)
        obj.source = source
        obj._lazy_shape = source.shape  # must be 2D
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.source = getattr(obj, 'source', None)
        self._lazy_shape = getattr(obj, '_lazy_shape', None)

    @property
    def shape(self):
        return self._lazy_shape  # e.g. (n_vars, n_vars)

    def __getitem__(self, idx):
        """
        Standard 2D indexing. 
        The result is a scalar, 1D, or 2D np.ndarray in memory.
        """
        data = self.source[idx]  # Read from disk
        data = np.asarray(data, dtype=self.dtype)
        return data

    def __repr__(self):
        return f"<LazyMatrix shape={self._lazy_shape}>"


class LazyAnnData(anndata.AnnData):
    """
    Subclass that can:
      1) Link `adata.X` to a 3D LazyTensor referencing `uns[track_key]`.
      2) Turn certain varp keys into 2D LazyMatrix references on disk.
    """

    def __init__(
        self,
        *args,
        track_key='tracks',        # The key in uns that will feed a 3D LazyTensor for X
        lazy_varp_keys=None,   # A list of keys in varp to store as 2D LazyMatrix
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._track_key = track_key
        self._lazy_varp_keys = set(lazy_varp_keys or [])

        # Post-process. If track_key is set, link X => uns[track_key]
        self._make_tracks_lazy()

        # If there are varp keys to be lazy, replace them
        self._make_varp_lazy()

    def _make_tracks_lazy(self):
        """
        If `track_key` is provided and exists in uns, we replace `self.X`
        with a 3D LazyTensor that references the on-disk dataset at uns[track_key].
        """
        if self.filename is None:
            return
        if self._track_key is None:
            return

        import h5py
        with h5py.File(self.filename, 'r') as f:
            if "uns" not in f:
                return
            if self._track_key in f["uns"]:
                dataset_path = f"uns/{self._track_key}"
                h5src = H5Source(self.filename, dataset_path)
                # e.g. shape=(N, N, M) or any logic you want:
                # Suppose the dataset is shape (N, M). We'll interpret that as (N, N, M):
                # shape = (self.shape[0], self.shape[1], h5src.shape[1])
                lazy_tens = LazyTensor(source=h5src, row_labels = list(self.obs_names), col_labels = list(self.var_names)) #shape=shape
                self.layers[self._track_key] = lazy_tens

    def _make_varp_lazy(self):
        """
        For each key in _lazy_varp_keys, replace varp[key] with a 2D LazyMatrix.
        """
        if self.filename is None:
            return
        import h5py
        with h5py.File(self.filename, 'r') as f:
            if "varp" not in f:
                return
            for key in self._lazy_varp_keys:
                if key in f["varp"]:
                    dataset_path = f"varp/{key}"
                    h5src = H5Source(self.filename, dataset_path)
                    lazy_mat = LazyMatrix(h5src)
                    self._varp[key] = lazy_mat

    def write(self, **kwargs):
        """
        In-place partial update, ensuring the file is in 'r+' mode.
        Example: rewriting `obs`, `var`, and `obsm`,
        but skipping huge `uns` or `varp`.
        """
        if self.filename is None:
            raise ValueError("No filename set. Can't do partial writes.")
    
        import h5py
        from anndata._io.h5ad import write_elem
    
        with h5py.File(self.filename, 'r+') as f:
            # Example: update `obs` if it's stored as a group
            if 'obs' in f and isinstance(f['obs'], h5py.Group):
                del f['obs']  # remove old
                write_elem(f,'obs',self.obs)
    
            # Example: update `var`
            if 'var' in f and isinstance(f['var'], h5py.Group):
                del f['var']
                write_elem(f,'var', self.var)
    
            # Overwrite `obsm` by iterating items manually
            if 'obsm' in f:
                del f['obsm']
            obsm_group = f.create_group('obsm')
            # Each entry in self.obsm is typically a 2D array or similar
            for key, arr in self.obsm.items():
                write_elem(obsm_group, key, arr)

def _write_raw_bigwigs_to_uns(
    h5ad_filename: str,
    consensus_peaks: pd.DataFrame,
    bigwig_files: list[str],
    track_key: str = "tracks",
    chunk_size: int = 1024,
) -> None:
    """
    Creates and populates an HDF5 dataset uns[track_key] in the .h5ad file with shape:
       (n_tracks, n_regions, max_length)

    - n_tracks = number of bigwig files
    - n_regions = number of rows in consensus_peaks
    - max_length = max(end - start) across all peaks

    Data is written row-by-row to avoid OOM. Each row is one bigWig file,
    and we iterate over the consensus_peaks in slices of size chunk_size.

    This is compatible with the LazyTensor class which expects row=track, col=region.

    Parameters
    ----------
    h5ad_filename : str
        The path to the existing .h5ad file (with minimal placeholder AnnData).
    consensus_peaks : pd.DataFrame
        DataFrame of consensus peaks. Must have at least columns 0,1,2 = chrom,start,end.
    bigwig_files : list[str]
        A sorted list of bigwig file paths (one per track).
    track_key : str
        The key in uns[track_key] to write the raw coverage dataset.
    chunk_size : int
        For memory reasons, we process the peaks in slices of this size.
    """
    # 1) figure out n_regions and max_len
    n_regions = len(consensus_peaks)
    max_len = 0
    # We assume columns [0,1,2] = (chrom, start, end), or you can adapt if you have named columns
    for row_i in range(n_regions):
        chrom = consensus_peaks.iat[row_i, 0]
        start = consensus_peaks.iat[row_i, 1]
        end   = consensus_peaks.iat[row_i, 2]
        length = int(end) - int(start)
        if length > max_len:
            max_len = length

    n_tracks = len(bigwig_files)

    # 2) open h5ad in r+ mode to create or overwrite uns[track_key]
    with h5py.File(h5ad_filename, "r+") as f:
        if "uns" not in f:
            f.create_group("uns")
        uns_grp = f["uns"]

        # remove old data if present
        if track_key in uns_grp:
            del uns_grp[track_key]

        # create dataset => shape (n_tracks, n_regions, max_len)
        # so row = track_i, col = region_i, last dim = coverage
        dset = uns_grp.create_dataset(
            track_key,
            shape=(n_tracks, n_regions, max_len),
            dtype="float32",
            chunks=(n_tracks, chunk_size, max_len),  # you can tune chunk shapes
            fillvalue=np.nan,
        )

        # 3) fill coverage row by row, chunk by chunk
        for track_i, bw_file in enumerate(bigwig_files):
            logger.info(f"Filling coverage for track {track_i+1}/{n_tracks}: {bw_file}")

            with pybigtools.open(bw_file, "r") as bw:
                cur_chroms = bw.chroms().keys()
                # read coverage in slices of size=chunk_size
                for start_idx in tqdm(range(0, n_regions, chunk_size)):
                    end_idx = min(start_idx + chunk_size, n_regions)
                    batch_size = end_idx - start_idx

                    # retrieve coverage for each region in [start_idx, end_idx)
                    batch_signals = []
                    local_max_len = 0
                    for reg_i in range(start_idx, end_idx):
                        chrom = consensus_peaks.iat[reg_i, 0]
                        start = consensus_peaks.iat[reg_i, 1]
                        end   = consensus_peaks.iat[reg_i, 2]
                        if chrom in cur_chroms:
                            signal = bw.values(
                                str(chrom),
                                int(start),
                                int(end),
                                # exact=False,
                                # missing=np.nan,
                                oob=np.nan,
                            )
                        else:
                            signal = np.zeros(int(end-start))
                        batch_signals.append(signal)
                        if len(signal) > local_max_len:
                            local_max_len = len(signal)

                    # clamp local_max_len by global max_len
                    if local_max_len > max_len:
                        local_max_len = max_len

                    # pad and write each coverage array
                    for offset, signal in enumerate(batch_signals):
                        region_i = start_idx + offset
                        length = len(signal)
                        if length > max_len:
                            length = max_len
                        # write to dset[track_i, region_i, :length]
                        dset[track_i, region_i, :length] = signal[:length].astype("float32")

    logger.info(
        f"Done writing raw coverage to uns[{track_key}] in {h5ad_filename}.\n"
        f"  Shape = (n_tracks={n_tracks}, n_regions={n_regions}, max_len={max_len})"
    )


def read_lazy_h5ad(
    filename: str | Path,
    mode: Literal["r", "r+"] = "r",
    track_key=None,
    lazy_varp_keys=None,
) -> LazyAnnData:
    """
    Reads an h5ad into a LazyAnnData.
      - Should have summary statistic X
      - Must have the tracks layer as uns
      - If track_key is set (e.g. "tracks"), we link adata.X => uns[track_key] as a 3D LazyTensor.
      - If lazy_varp_keys is set, each varp[key] is replaced by a 2D LazyMatrix.
    """
    import h5py

    filename = str(filename)
    init_kwargs = {
        "filename": filename,
        "filemode": mode,
        "track_key": track_key,
        "lazy_varp_keys": lazy_varp_keys,
    }

    with h5py.File(filename, mode) as f:
        from anndata._io.h5ad import read_elem, read_dataframe
        attributes = ["obsm"]
        df_attributes = ["obs", "var"]

        # Minimal logic from anndata
        if "encoding-type" in f.attrs:
            attributes.extend(df_attributes)
        else:
            for k in df_attributes:
                if k in f:
                    init_kwargs[k] = read_dataframe(f[k])

        for attr in attributes:
            if attr in f:
                init_kwargs[attr] = read_elem(f[attr])

    # Now create the LazyAnnData, which calls _make_tracks_lazy and _make_varp_lazy
    adata = LazyAnnData(**init_kwargs)
    return adata

def filter_and_adjust_chromosome_data(
    peaks: pd.DataFrame,
    chrom_sizes: dict,
    max_shift: int = 0,
    chrom_col: str = "chr",
    start_col: str = "start",
    end_col: str = "end",
    MIN_POS: int = 0,
) -> pd.DataFrame:
    """
    Expand each peak by `max_shift` on both sides if possible.
    If the peak is near the left edge, the leftover shift is added to the right side;
    if near the right edge, leftover shift is added to the left side.

    Returns a DataFrame where each row is expanded (unless blocked by chromosome edges).
    Rows with an unknown chromosome (i.e., not found in chrom_sizes) are dropped.

    Example:
      If a row is: chr1, start=0, end=2114, max_shift=50
      => desired new length = 2114 + 2*50 = 2214
      => final row: (chr1, 0, 2214)
    """

    # 1) Map each row's chromosome to its size
    #    Rows with missing chrom sizes become NaN → we drop them.
    peaks["_chr_size"] = peaks[chrom_col].map(chrom_sizes)
    peaks = peaks.dropna(subset=["_chr_size"]).copy()
    peaks["_chr_size"] = peaks["_chr_size"].astype(int)
    
    # Convert to arrays for fast vectorized arithmetic
    starts = peaks[start_col].to_numpy(dtype=int)
    ends = peaks[end_col].to_numpy(dtype=int)
    chr_sizes_arr = peaks["_chr_size"].to_numpy(dtype=int)

    # Original length
    orig_length = ends - starts
    desired_length = orig_length + 2 * max_shift

    # 2) Temporarily shift left by max_shift
    new_starts = starts - max_shift
    new_ends = new_starts + desired_length  # (so that final length = desired_length)

    # 3) If new_start < MIN_POS, shift leftover to the right
    cond_left_edge = new_starts < MIN_POS
    # How far below MIN_POS did we go?
    shift_needed = MIN_POS - new_starts[cond_left_edge]  # positive number
    new_starts[cond_left_edge] = MIN_POS
    new_ends[cond_left_edge] += shift_needed

    # 4) If new_end > chr_size, shift leftover to the left
    cond_right_edge = new_ends > chr_sizes_arr
    # How far beyond chromosome size did we go?
    shift_needed = new_ends[cond_right_edge] - chr_sizes_arr[cond_right_edge]
    new_ends[cond_right_edge] = chr_sizes_arr[cond_right_edge]
    new_starts[cond_right_edge] -= shift_needed

    # 5) If shifting back on the left made new_start < MIN_POS again, clamp it.
    cond_left_clamp = new_starts < MIN_POS
    new_starts[cond_left_clamp] = MIN_POS

    # Assign back to DataFrame
    peaks[start_col] = new_starts
    peaks[end_col] = new_ends

    peaks.drop(columns=["_chr_size"], inplace=True)

    return peaks


def import_bigwigs_raw(
    bigwigs_folder: PathLike,
    regions_file: PathLike,
    h5ad_path: PathLike,
    target_region_width: int | None,
    chromsizes_file: PathLike | None = None,
    genome: Genome | None = None,
    max_stochastic_shift: int = 0,
    chunk_size: int = 512,
) -> AnnData:
    """
    Import bigWig files and consensus regions BED file into AnnData format.

    This format is required to be able to train a peak prediction model.
    The bigWig files target values are calculated for each region and and imported into an AnnData object,
    with the bigWig file names as .obs and the consensus regions as .var.
    Optionally, the target region width can be specified to extract values from a wider/narrower region around the consensus region,
    where the original region will still be used as the index.
    This is often useful to extract sequence information around the actual peak region.

    Parameters
    ----------
    bigwigs_folder
        Folder name containing the bigWig files.
    regions_file
        File name of the consensus regions BED file.
    file_path
        Path where the anndata file will be backed
    chromsizes_dict
        Chromsizes dictionary. Used for checking if the new regions are within the chromosome boundaries.
        If not provided, will look for a registered genome object.
    target_region_width
        Width of region that the bigWig target value will be extracted from. If None, the
        consensus region width will be used.

    Returns
    -------
    AnnData object with bigWigs as rows and peaks as columns.

    Example
    -------
    >>> anndata = crested.import_bigwigs(
    ...     bigwigs_folder="path/to/bigwigs",
    ...     regions_file="path/to/peaks.bed",
    ...     chromsizes_file="path/to/chrom.sizes",
    ...     target="max",
    ...     target_region_width=500,
    ... )
    """
    bigwigs_folder = Path(bigwigs_folder)
    regions_file = Path(regions_file)

    # Input checks
    if not bigwigs_folder.is_dir():
        raise FileNotFoundError(f"Directory '{bigwigs_folder}' not found")
    if not regions_file.is_file():
        raise FileNotFoundError(f"File '{regions_file}' not found")
    if chromsizes_file is not None:
        chromsizes_dict = _read_chromsizes(chromsizes_file)
    if genome is not None:
        chromsizes_dict = genome.chrom_sizes
    
    # Read consensus regions BED file and filter out regions not within chromosomes
    _check_bed_file_format(regions_file)
    consensus_peaks = _read_consensus_regions(regions_file, chromsizes_dict)
    consensus_peaks = filter_and_adjust_chromosome_data(consensus_peaks, chromsizes_dict, max_shift=max_stochastic_shift)
    shifted_width = (target_region_width+2*max_stochastic_shift)
    consensus_peaks = consensus_peaks.loc[(consensus_peaks['end']-consensus_peaks['start']) == shifted_width,:]
    
    bw_files = []
    chrom_set = set([])
    for file in os.listdir(bigwigs_folder):
        file_path = os.path.join(bigwigs_folder, file)
        try:
            # Validate using pyBigTools (add specific validation if available)
            bw = pybigtools.open(file_path, "r")
            chrom_set = chrom_set | set(bw.chroms().keys())
            bw_files.append(file_path)
            bw.close()
        except ValueError:
            pass
        except pybigtools.BBIReadError:
            pass


    consensus_peaks = consensus_peaks.loc[consensus_peaks['chr'].isin(chrom_set),:] 
    
    bw_files = sorted(bw_files)
    if not bw_files:
        raise FileNotFoundError(f"No valid bigWig files found in '{bigwigs_folder}'")

    # Process bigWig files in parallel and collect the results
    logger.info(f"Extracting values from {len(bw_files)} bigWig files...")
    all_results = []

    # Prepare obs and var for AnnData
    obs_df = pd.DataFrame(
        data={"file_path": bw_files},
        index=[
            os.path.basename(file).rpartition(".")[0].replace(".", "_")
            for file in bw_files
        ],
    )
    var_df = consensus_peaks.set_index("region")
    
    # Create AnnData object
    adata = ad.AnnData(X = csr_matrix((obs_df.shape[0],var_df.shape[0])), obs=obs_df, var=var_df)
    adata.write_h5ad(h5ad_path)

    _write_raw_bigwigs_to_uns(
        h5ad_filename=h5ad_path,
        consensus_peaks=consensus_peaks,
        bigwig_files=bw_files,
        track_key="tracks",
        chunk_size=chunk_size,
    )
    
    lazy_adata = read_lazy_h5ad(filename=h5ad_path, mode="r+" , track_key="tracks")#, lazy_varp_keys=[varp_keys]
    return lazy_adata




    
