"""I/O functions for importing topics and bigWigs into AnnData objects."""

from __future__ import annotations

import os
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


def _sort_topic_files(filename: str):
    """Sorts topic files.

    Prioritizes numeric extraction from filenames of the format 'Topic_X.bed' (X=int).
    Other filenames are sorted alphabetically, with 'Topic_' files coming last if numeric extraction fails.
    """
    filename = Path(filename)
    parts = filename.stem.split("_")

    if len(parts) > 1:
        try:
            return (False, int(parts[1]))
        except ValueError:
            # If the numeric part is not an integer, handle gracefully
            return (True, filename.stem)

    # Return True for the first element to sort non-'Topic_X' filenames alphabetically after 'Topic_X'
    return (
        True,
        filename.stem,
    )


def _read_chromsizes(chromsizes_file: PathLike) -> dict[str, int]:
    """Read chromsizes file into a dictionary."""
    chromsizes = pd.read_csv(
        chromsizes_file, sep="\t", header=None, names=["chr", "size"]
    )
    chromsizes_dict = chromsizes.set_index("chr")["size"].to_dict()
    return chromsizes_dict


def _extract_values_from_bigwig(bw_file, bed_file, target, target_region_width):
    """Extract target values from a bigWig file for regions specified in a BED file."""
    if isinstance(bed_file, Path):
        bed_file = str(bed_file)
    if isinstance(bw_file, Path):
        bw_file = str(bw_file)

    if target == "mean":
        values = list(
            pybigtools.bigWigAverageOverBed(bw_file, bed=bed_file, names=None)
        )
    elif target == "max":
        values = list(
            pybigtools.bigWigAverageOverBed(bw_file, bed=bed_file, names=None)
        )
    elif target == "count":
        values = list(
            pybigtools.bigWigAverageOverBed(bw_file, bed=bed_file, names=None)
        )
    elif target == "logcount":
        values = list(
            pybigtools.bigWigAverageOverBed(bw_file, bed=bed_file, names=None)
        )
    else:
        raise ValueError(f"Unsupported target '{target}'")

    return values


def _read_consensus_regions(
    regions_file: PathLike, chromsizes_file: PathLike | None = None
) -> pd.DataFrame:
    """Read consensus regions BED file and filter out regions not within chromosomes."""
    consensus_peaks = pd.read_csv(
        regions_file, sep="\t", header=None, usecols=[0, 1, 2]
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

    return consensus_peaks


def _create_temp_bed_file(
    consensus_peaks: pd.DataFrame, target_region_width: int
) -> str:
    """Adjust consensus regions to a target width and create a temporary BED file."""
    adjusted_peaks = consensus_peaks.copy()
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


def import_topics(
    topics_folder: PathLike,
    regions_file: PathLike,
    chromsizes_file: PathLike | None = None,
    topics_subset: list | None = None,
    remove_empty_regions: bool = True,
    compress: bool = False,
) -> AnnData:
    """
    Import topic and consensus regions BED files into AnnData format.

    This format is required to be able to train a topic prediction model.
    The topic and consensus regions are the outputs from running pycisTopic
    (https://pycistopic.readthedocs.io/en/latest/) on your data.
    The result is an AnnData object with topics as rows and consensus region as columns,
    with binary values indicating whether a region is present in a topic.

    Example
    -------
    >>> anndata = crested.import_topics(
    ...     topics_folder="path/to/topics",
    ...     regions_file="path/to/regions.bed",
    ...     chromsizes_file="path/to/chrom.sizes",
    ...     topics_subset=["Topic_1", "Topic_2"],
    ... )

    Parameters
    ----------
    topics_folder
        Folder path containing the topic BED files.
    regions_file
        File path of the consensus regions BED file.
    topics_subset
        List of topics to include in the AnnData object. If None, all topics
        will be included.
        Topics should be named after the topics file name without the extension.
    chromsizes_file
        File path of the chromsizes file.
    remove_empty_regions
        Remove regions that are not open in any topic.
    compress
        Compress the AnnData.X matrix. If True, the matrix will be stored as
        a sparse matrix. If False, the matrix will be stored as a dense matrix.

        WARNING: Compressing the matrix currently makes training very slow and is never recommended.
        We're still investigating a way around.

    Returns
    -------
    AnnData object with topics as rows and peaks as columns.
    """
    topics_folder = Path(topics_folder)
    regions_file = Path(regions_file)

    # Input checks
    if not topics_folder.is_dir():
        raise FileNotFoundError(f"Directory '{topics_folder}' not found")
    if not regions_file.is_file():
        raise FileNotFoundError(f"File '{regions_file}' not found")
    if chromsizes_file is not None:
        chromsizes_file = Path(chromsizes_file)
        if not chromsizes_file.is_file():
            raise FileNotFoundError(f"File '{chromsizes_file}' not found")
    if chromsizes_file is None:
        logger.warning(
            "Chromsizes file not provided. Will not check if regions are within chromosomes",
            stacklevel=1,
        )
    if topics_subset is not None:
        for topic in topics_subset:
            if not any(topics_folder.glob(f"{topic}.bed")):
                raise FileNotFoundError(
                    f"Topic '{topic}' not found in '{topics_folder}'"
                )

    # Read consensus regions BED file and filter out regions not within chromosomes
    consensus_peaks = _read_consensus_regions(regions_file, chromsizes_file)

    binary_matrix = pd.DataFrame(0, index=[], columns=consensus_peaks["region"])
    topic_file_paths = []

    # Which topic regions are present in the consensus regions
    logger.info(f"Reading topics from {topics_folder}...")
    for topic_file in sorted(topics_folder.glob("*.bed"), key=_sort_topic_files):
        topic_name = topic_file.stem
        if topics_subset is None or topic_name in topics_subset:
            topic_peaks = pd.read_csv(
                topic_file, sep="\t", header=None, usecols=[0, 1, 2]
            )
            topic_peaks["region"] = (
                topic_peaks[0].astype(str)
                + ":"
                + topic_peaks[1].astype(str)
                + "-"
                + topic_peaks[2].astype(str)
            )

            # Create binary row for the current topic
            binary_row = binary_matrix.columns.isin(topic_peaks["region"]).astype(int)
            binary_matrix.loc[topic_name] = binary_row
            topic_file_paths.append(str(topic_file))

    ann_data = AnnData(
        binary_matrix,
    )

    ann_data.obs["file_path"] = topic_file_paths
    ann_data.obs["n_open_regions"] = ann_data.X.sum(axis=1)
    ann_data.var["n_topics"] = ann_data.X.sum(axis=0)
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
    topics_no_open_regions = ann_data.obs[ann_data.obs["n_open_regions"] == 0]
    if not topics_no_open_regions.empty:
        raise ValueError(
            f"Topics {topics_no_open_regions.index} have 0 open regions in the consensus peaks"
        )
    regions_no_topics = ann_data.var[ann_data.var["n_topics"] == 0]
    if not regions_no_topics.empty:
        if remove_empty_regions:
            logger.warning(
                f"{len(regions_no_topics.index)} consensus regions are not open in any topic. Removing them from the AnnData object. Disable this behavior by setting 'remove_empty_regions=False'",
            )
            ann_data = ann_data[:, ann_data.var["n_topics"] > 0]

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

    Example
    -------
    >>> anndata = crested.import_peaks(
    ...     bigwigs_folder="path/to/bigwigs",
    ...     regions_file="path/to/peaks.bed",
    ...     chromsizes_file="path/to/chrom.sizes",
    ...     target="max",
    ...     target_region_width=500,
    ... )

    Parameters
    ----------
    bigwigs_folder
        Folder name containing the bigWig files.
    regions_file
        File name of the consensus regions BED file.
    chromsizes_file
        File name of the chromsizes file.
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
    """
    bigwigs_folder = Path(bigwigs_folder)
    regions_file = Path(regions_file)

    # Input checks
    if not bigwigs_folder.is_dir():
        raise FileNotFoundError(f"Directory '{bigwigs_folder}' not found")
    if not regions_file.is_file():
        raise FileNotFoundError(f"File '{regions_file}' not found")
    if chromsizes_file is not None:
        chromsizes_file = Path(chromsizes_file)
        if not chromsizes_file.is_file():
            raise FileNotFoundError(f"File '{chromsizes_file}' not found")
    if chromsizes_file is None:
        logger.warning(
            "Chromsizes file not provided. Will not check if regions are within chromosomes",
            stacklevel=1,
        )

    # Read consensus regions BED file and filter out regions not within chromosomes
    consensus_peaks = _read_consensus_regions(regions_file, chromsizes_file)

    if target_region_width is not None:
        bed_file = _create_temp_bed_file(consensus_peaks, target_region_width)
    else:
        bed_file = regions_file

    bw_files = [
        os.path.join(bigwigs_folder, file)
        for file in os.listdir(bigwigs_folder)
        if file.endswith(".bw")
    ]
    bw_files = sorted(bw_files)

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
                target_region_width,
            )
            for bw_file in bw_files
        ]
        for future in futures:
            all_results.append(future.result())

    if target_region_width is not None:
        os.remove(bed_file)

    data_matrix = np.array(all_results)

    # Create DataFrame for AnnData
    df = pd.DataFrame(
        data_matrix,
        columns=consensus_peaks["region"],
        index=[os.path.basename(file).split(".")[0] for file in bw_files],
    )

    # Create AnnData object
    ann_data = ad.AnnData(df)

    ann_data.obs["file_path"] = bw_files
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
    regions_no_values = ann_data.var[ann_data.X.sum(axis=0) == 0]
    if not regions_no_values.empty:
        logger.warning(
            f"{len(regions_no_values.index)} consensus regions have no values in any bigWig file",
        )

    return ann_data
