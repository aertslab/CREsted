"""I/O functions for importing beds and bigWigs into AnnData objects."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping, Sequence
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

from crested import _conf as conf


def _sort_files(filename: str | PathLike):
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


def _read_chromsizes(chromsizes_file: str | PathLike) -> dict[str, int]:
    """Read chromsizes file into a dictionary."""
    chromsizes = pd.read_csv(
        chromsizes_file, sep="\t", header=None, names=["chr", "size"]
    )
    chromsizes_dict = chromsizes.set_index("chr")["size"].to_dict()
    return chromsizes_dict


def _extract_values_from_bigwig(
    bw_file: str | PathLike, bed_file: str | PathLike, target: str
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
        # Make sure all content is written to temporary BED file.
        temp_bed_file.file.flush()

    total_bed_entries = idx + 1
    bed_entries_to_keep_idx = np.array(bed_entries_to_keep_idx, np.intp)

    # Warn if we filtered out a significant amount of regions
    if (len(bed_entries_to_keep_idx) / total_bed_entries) < 0.25:
        logger.warning(
            f"{(1 - len(bed_entries_to_keep_idx) / total_bed_entries) * 100:.2f}% of BED regions' chromosomes did not match chromosomes in BigWig file {bw_file} and were filtered out."
        )

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

    # Remove temporary BED file.
    temp_bed_file.close()

    # Check for negative values
    if any(values < 0):
        logger.warning(
            f"Peak heights from bigwig {bw_file} contain negative values, which most models in CREsted don't support. Proceed with caution."
        )

    if np.isnan(values).all():
        raise ValueError(
            f"All read-in values are NaNs. Your region chromosomes most likely don't match your bigwig dataset for bigWig {bw_file}."
        )

    if values.shape[0] != total_bed_entries:
        # Set all values for BED entries for which the chromosome was not in in the bigWig file to NaN.
        all_values = np.full(total_bed_entries, np.nan, dtype=np.float32)
        all_values[bed_entries_to_keep_idx] = values
        return all_values
    else:
        return values


def _extract_tracks_from_bigwig(
    bw_file: str | PathLike,
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

    # Get region total chromosome set
    coordinate_chroms = {region[0] for region in coordinates}

    # Open the bigWig file
    with pybigtools.open(bw_file, "r") as bw:
        # Check chromosome match
        bw_chroms = set(bw.chroms().keys())
        if all(coord_chrom not in bw_chroms for coord_chrom in coordinate_chroms):
            raise ValueError(
                f"None of the data chromosomes match BigWig {bw_file}'s chromosomes.",
                f"Data chromosomes: {coordinate_chroms}",
                f"BigWig chromosomes: {bw_chroms}",
            )
        # Read out values
        results = np.empty(
            (len(coordinates), binned_length), dtype="float64"
        )  # pybigtools returns values in float64
        for i, region in enumerate(coordinates):
            chrom, start, end = region
            # Extract values
            bw.values(
                chrom,
                start,
                end,
                bins=bins,
                summary=target,
                exact=exact,
                missing=missing,
                oob=oob,
                arr=results[i, ...],
            )

    if (results < 0).any():
        logger.warning(
            f"Tracks from bigWig {bw_file} contain negative values, which most models in CREsted don't support. Proceed with caution."
        )

    if np.isnan(results).all():
        raise ValueError(
            f"All read-in values are NaNs. Your region chromosomes most likely don't match your bigwig dataset for bigWig {bw_file}."
        )

    return results


def _read_consensus_regions(
    regions_file: str | PathLike, chromsizes_file: str | PathLike | None = None
) -> pd.DataFrame:
    """Read consensus regions BED file and filter out regions not within chromosomes."""
    if chromsizes_file is not None:
        chromsizes_file = Path(chromsizes_file)
        if not chromsizes_file.is_file():
            raise FileNotFoundError(f"File '{chromsizes_file}' not found")
    if chromsizes_file is None and not conf.genome:
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
    consensus_peaks["region"] = (
        consensus_peaks[0].astype(str)
        + ":"
        + consensus_peaks[1].astype(str)
        + "-"
        + consensus_peaks[2].astype(str)
    )
    if chromsizes_file:
        chromsizes_dict = _read_chromsizes(chromsizes_file)
    elif conf.genome:
        chromsizes_dict = conf.genome.chrom_sizes
    else:
        return consensus_peaks

    valid_mask = consensus_peaks.apply(
        lambda row: row[0] in chromsizes_dict
        and row[1] >= 0
        and row[2] <= chromsizes_dict[row[0]],
        axis=1,
    )
    consensus_peaks_filtered = consensus_peaks[valid_mask]

    if len(consensus_peaks) != len(consensus_peaks_filtered):
        logger.warning(
            f"Filtered {len(consensus_peaks) - len(consensus_peaks_filtered)} consensus regions (not within chromosomes)",
        )
    if len(consensus_peaks_filtered) == 0:
        raise ValueError(
            f"None of the consensus regions in {regions_file} fell within known chromosomes. Your registered genome likely doesn't match your consensus region's chromosome names (e.g. chrI vs I)."
        )
    return consensus_peaks_filtered


def _create_temp_bed_file(
    consensus_peaks: pd.DataFrame, target_region_width: int | None
) -> str:
    """Adjust consensus regions to a target width and create a temporary BED file."""
    adjusted_peaks = consensus_peaks.copy()
    if target_region_width:
        adjusted_peaks[1] = adjusted_peaks.apply(
            lambda row: max(0, row[1] - (target_region_width - (row[2] - row[1])) // 2),
            axis=1,
        )
        adjusted_peaks[2] = adjusted_peaks[1] + target_region_width
    adjusted_peaks[1] = adjusted_peaks[1].astype(int)
    adjusted_peaks[2] = adjusted_peaks[2].astype(int)

    # Create a temporary BED file
    temp_bed_file = tempfile.NamedTemporaryFile(delete=False, mode="w+t").name
    adjusted_peaks.to_csv(temp_bed_file, sep="\t", header=False, index=False)
    return temp_bed_file


def _check_bed_file_format(bed_file: str | PathLike) -> None:
    """Check if the BED file is in the correct format."""
    with open(bed_file) as f:
        first_line = f.readline().strip()
    # check if at least three columns are found
    if len(first_line.split("\t")) < 3:
        raise ValueError(
            f"BED file '{bed_file}' is not in the correct format. Expected at least three tab-seperated columns."
        )
    pattern = r".*\t\d+\t\d+.*"
    if not re.match(pattern, first_line):
        raise ValueError(
            f"BED file '{bed_file}' is not in the correct format. Expected columns 2 and 3 to contain integers."
        )


def import_beds(
    beds_folder: list[str] | dict[str, str] | str | PathLike,
    regions_file: str | PathLike | None = None,
    chromsizes_file: str | PathLike | None = None,
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
        List of bed file paths, dict of bed paths with class name keys, or folder path containing the bed files.
        If a path to a folder, assumed all bed files have the .bed extension.
    regions_file
        File path of the consensus regions BED file to use as columns in the AnnData object.
        If None, the regions will be extracted from the files.
    classes_subset
        List of classes to include in the AnnData object when providing a folder to read. If None, all files will be included.
        Classes should be named after the file name without the extension.
    chromsizes_file
        File path of the chromsizes file.  Used for checking if the new regions are within the chromosome boundaries.
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
    ...     chromsizes_file="path/to/chrom.sizes",
    ...     classes_subset=["Topic_1", "Topic_2"],
    ... )
    """
    regions_file = Path(regions_file) if regions_file else None

    # Input checks
    if (regions_file is not None) and (not regions_file.is_file()):
        raise FileNotFoundError(f"File '{regions_file}' not found")

    # Get list of files with their associated names
    if isinstance(beds_folder, str) or isinstance(beds_folder, PathLike):
        beds_folder = Path(beds_folder)
        if not beds_folder.is_dir():
            raise FileNotFoundError(
                f"Directory '{beds_folder}' not found or not a directory."
            )
        bed_files = {}
        # Find the files that look like beds
        bed_files = {
            bed_file.stem: bed_file
            for bed_file in sorted(beds_folder.glob("*.bed"), key=_sort_files)
        }
        if len(bed_files) == 0:
            raise FileNotFoundError(f"No bed files found in '{beds_folder}'")
        if classes_subset is not None:
            bed_files = {
                class_name: bed_path
                for class_name, bed_path in bed_files.items()
                if class_name in classes_subset
            }
            if len(bed_files) == 0:
                raise FileNotFoundError(
                    f"No bed files matching classes_subset classes ({classes_subset}) found in beds_folder ({beds_folder})."
                )
    elif isinstance(beds_folder, Sequence):
        if classes_subset is not None:
            raise ValueError(
                "Argument classes_subset only works if beds_folder is a folder. To subset classes, simply remove them from your list of files."
            )
        bed_files = {
            os.path.basename(file_path).rpartition(".")[0].replace(".", "_"): file_path
            for file_path in beds_folder
        }
    elif isinstance(beds_folder, Mapping):
        if classes_subset is not None:
            raise ValueError(
                "Argument classes_subset only works if beds_folder is a folder. To subset classes, simply remove them from your dict of files."
            )
        bed_files = dict(beds_folder)
    else:
        raise ValueError(
            "Could not recognise argument beds_folder as a path, a list of paths, or a dict of paths."
        )

    # Check if all files exist
    for file_path in bed_files.values():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find file {file_path}.")

    if regions_file:
        # Read consensus regions BED file and filter out regions not within chromosomes
        _check_bed_file_format(regions_file)
        consensus_peaks = _read_consensus_regions(regions_file, chromsizes_file)

        binary_matrix = pd.DataFrame(0, index=[], columns=consensus_peaks["region"])

        # Which regions are present in the consensus regions
        logger.info(
            f"Reading bed files from {beds_folder} and using {regions_file} as var_names..."
        )
        for class_name, bed_file in bed_files.items():
            class_regions = pd.read_csv(
                bed_file, sep="\t", header=None, usecols=[0, 1, 2]
            )
            class_regions["region"] = (
                class_regions[0].astype(str)
                + ":"
                + class_regions[1].astype(str)
                + "-"
                + class_regions[2].astype(str)
            )

            # Create binary row for the current topic
            binary_row = binary_matrix.columns.isin(class_regions["region"]).astype(int)
            binary_matrix.loc[class_name] = binary_row

    # else, get regions from the bed files
    else:
        all_regions = set()

        # Collect all regions from the BED files
        logger.info(
            f"Reading bed files from {beds_folder} without consensus regions..."
        )
        # Construct total set of region names
        for _, bed_file in bed_files.values():
            _check_bed_file_format(bed_file)
            class_regions = pd.read_csv(
                bed_file, sep="\t", header=None, usecols=[0, 1, 2]
            )
            class_regions["region"] = (
                class_regions[0].astype(str)
                + ":"
                + class_regions[1].astype(str)
                + "-"
                + class_regions[2].astype(str)
            )
            all_regions.update(class_regions["region"].tolist())

        # Save presence for each file
        # Convert set to sorted list
        all_regions = sorted(all_regions, key=_custom_region_sort)
        binary_matrix = pd.DataFrame(0, index=[], columns=all_regions)

        # Populate the binary matrix
        for class_name, bed_file in bed_files.values():
            class_regions = pd.read_csv(
                bed_file, sep="\t", header=None, usecols=[0, 1, 2]
            )
            class_regions["region"] = (
                class_regions[0].astype(str)
                + ":"
                + class_regions[1].astype(str)
                + "-"
                + class_regions[2].astype(str)
            )
            binary_row = binary_matrix.columns.isin(class_regions["region"]).astype(int)
            binary_matrix.loc[class_name] = binary_row

    adata = AnnData(
        binary_matrix,
    )

    adata.obs["file_path"] = list(bed_files.values())
    adata.obs["n_open_regions"] = adata.X.sum(axis=1)
    adata.var["n_classes"] = adata.X.sum(axis=0)
    adata.var["chr"] = adata.var.index.str.split(":").str[0]
    adata.var["start"] = (
        adata.var.index.str.split(":").str[1].str.split("-").str[0]
    ).astype(int)
    adata.var["end"] = (
        adata.var.index.str.split(":").str[1].str.split("-").str[1]
    ).astype(int)

    if compress:
        adata.X = csr_matrix(adata.X)

    # Output checks
    classes_no_open_regions = adata.obs[adata.obs["n_open_regions"] == 0]
    if not classes_no_open_regions.empty:
        raise ValueError(
            f"{classes_no_open_regions.index} have 0 open regions in the consensus peaks"
        )
    regions_no_classes = adata.var[adata.var["n_classes"] == 0]
    if not regions_no_classes.empty:
        if remove_empty_regions:
            logger.warning(
                f"{len(regions_no_classes.index)} consensus regions are not open in any class. Removing them from the AnnData object. Disable this behavior by setting 'remove_empty_regions=False'",
            )
            adata = adata[:, adata.var["n_classes"] > 0].copy()

    return adata


def import_bigwigs(
    bigwigs_folder: list[str] | dict[str, str] | str | PathLike,
    regions_file: str | PathLike,
    chromsizes_file: str | PathLike | None = None,
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
        List of bigWig file paths, dict of bigWig paths with class name keys, or folder path containing the bigWig files.
    regions_file
        File name of the consensus regions BED file.
    chromsizes_file
        File name of the chromsizes file. Used for checking if the new regions are within the chromosome boundaries.
        If not provided, will look for a registered genome object.
    target
        Target value to extract from bigwigs. Can be 'mean', 'max', 'count', or 'logcount'
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
    # Gather bigwig paths
    if isinstance(bigwigs_folder, str) or isinstance(bigwigs_folder, PathLike):
        bigwigs_folder = Path(bigwigs_folder)
        if not bigwigs_folder.is_dir():
            raise FileNotFoundError(
                f"Directory '{bigwigs_folder}' not found or not a directory."
            )
        bw_files = {}
        # Find the files that look like bigwigs
        for file in os.listdir(bigwigs_folder):
            file_path = os.path.join(bigwigs_folder, file)
            try:
                # Validate using pyBigTools (add specific validation if available)
                bw = pybigtools.open(file_path, "r")
                bw_name = (
                    os.path.basename(file_path).rpartition(".")[0].replace(".", "_")
                )
                bw_files[bw_name] = file_path
                bw.close()
            except ValueError:
                pass
            except pybigtools.BBIReadError:
                pass
        if len(bw_files) == 0:
            raise FileNotFoundError(
                f"No valid bigWig files found in '{bigwigs_folder}'"
            )
    elif isinstance(bigwigs_folder, Sequence):
        bw_files = {
            os.path.basename(file_path).rpartition(".")[0].replace(".", "_"): file_path
            for file_path in bigwigs_folder
        }
    elif isinstance(bigwigs_folder, Mapping):
        bw_files = dict(bigwigs_folder)
    else:
        raise ValueError(
            "Could not recognise argument bigwigs_folder as a path, a list of paths, or a dict of paths."
        )

    for file_path in bw_files.values():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find file {file_path}.")
        with pybigtools.open(file_path, "r") as bw:
            if not bw.is_bigwig:
                raise pybigtools.BBIReadError(
                    f"File {file_path} does not seem to be a bigwig file."
                )

    bw_files = {bw_name: bw_files[bw_name] for bw_name in sorted(bw_files)}

    # Process regions file
    regions_file = Path(regions_file)
    if not regions_file.is_file():
        raise FileNotFoundError(f"File '{regions_file}' not found")

    # Read consensus regions BED file and filter out regions not within chromosomes
    _check_bed_file_format(regions_file)
    consensus_peaks = _read_consensus_regions(regions_file, chromsizes_file)

    bed_file = _create_temp_bed_file(consensus_peaks, target_region_width)

    # Process bigWig files in parallel and collect the results
    logger.info(f"Extracting values from {len(bw_files)} bigWig files...")
    all_results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _extract_values_from_bigwig,
                bw_file,
                bed_file,
                target,
            )
            for bw_file in bw_files.values()
        ]
        for future in futures:
            all_results.append(future.result())

    os.remove(bed_file)

    data_matrix = np.vstack(all_results)

    # Prepare obs and var for AnnData
    obs_df = pd.DataFrame(
        index=list(bw_files.keys()),
        data={"file_path": list(bw_files.values())},
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

    if target_region_width:
        var_df["target_start"] = var_df.apply(
            lambda row: max(
                0,
                row["start"] - (target_region_width - (row["end"] - row["start"])) // 2,
            ),
            axis=1,
        )
        var_df["target_end"] = var_df["target_start"] + target_region_width

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
