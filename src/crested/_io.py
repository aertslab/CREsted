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
        # Make sure all content is written to temporary BED file.
        temp_bed_file.file.flush()

    total_bed_entries = idx + 1
    bed_entries_to_keep_idx = np.array(bed_entries_to_keep_idx, np.intp)

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

    if values.shape[0] != total_bed_entries:
        # Set all values for BED entries for which the chromosome was not in in the bigWig file to NaN.
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
    regions_file: PathLike, chromsizes_file: PathLike | None = None
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
    chromsizes_file: PathLike | None = None,
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
        consensus_peaks = _read_consensus_regions(regions_file, chromsizes_file)

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
    chromsizes_file: PathLike | None = None,
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

    bed_file = _create_temp_bed_file(consensus_peaks, target_region_width)

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
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _extract_values_from_bigwig,
                bw_file,
                bed_file,
                target,
            )
            for bw_file in bw_files
        ]
        for future in futures:
            all_results.append(future.result())

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
