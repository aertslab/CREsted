from __future__ import annotations

from os import PathLike
from pathlib import Path

import pandas as pd
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


def import_topics(
    topics_folder: PathLike,
    regions_file: PathLike,
    chromsizes_file: PathLike | None = None,
    topics_subset: list | None = None,
    remove_empty_regions: bool = True,
    compress: bool = True,
) -> AnnData:
    """
    Import topic and consensus regions BED files into AnnData format.

    This format is required to be able to train a DeepTopic model.
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

    # Read consensus regions BED file
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

    # Check if regions are within chromosomes
    if chromsizes_file is not None:
        chromsizes = pd.read_csv(
            chromsizes_file, sep="\t", header=None, names=["chr", "size"]
        )
        chromsizes_dict = chromsizes.set_index("chr")["size"].to_dict()
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
        consensus_peaks = consensus_peaks_filtered

    binary_matrix = pd.DataFrame(0, index=[], columns=consensus_peaks["region"])
    topic_file_paths = []

    # Which topic regions are present in the consensus regions
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


def import_peaks(
    bigwigs_folder: PathLike,
    regions_file: PathLike,
    chromsizes_file: PathLike,
    target: str = "mean",
    target_region_width: int | None = None,
    compress: bool = True,
) -> AnnData:
    """
    Import bigWig files and consensus regions BED file into AnnData format.

    This format is required to be able to train a peak prediction model.
    The bigWig files are the inputs to the model, and the consensus regions
    are the targets. The result is an AnnData object with bigWigs as rows and
    peaks as columns, with the bigWig values at each peak.

    Example
    -------
    >>> anndata = crested.import_peaks(
    ...     bigwigs_folder="path/to/bigwigs",
    ...     regions_file="path/to/peaks.bed",
    ...     chromsizes_file="path/to/chrom.sizes,
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
        Width of region that the target value will be extracted from. If None, the
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
    chromsizes_file = Path(chromsizes_file)

    # Input checks
    if not bigwigs_folder.is_dir():
        raise FileNotFoundError(f"Directory '{bigwigs_folder}' not found")
    if not regions_file.is_file():
        raise FileNotFoundError(f"File '{regions_file}' not found")
    if not chromsizes_file.is_file():
        raise FileNotFoundError(f"File '{chromsizes_file}' not found")

    raise NotImplementedError
