from __future__ import annotations

import warnings
from os import PathLike
from pathlib import Path

import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix


def import_topics(
    topics_folder: PathLike,
    peaks_file: PathLike,
    topics_subset: list | None = None,
    compress: bool = True,
) -> AnnData:
    """
    Import topic and consensus regions BED files into AnnData format.

    This format is required to be able to train a DeepTopic model.
    The topic and consensus regions are the outputs from running pycisTopic
    (https://pycistopic.readthedocs.io/en/latest/) on your data.
    The result is an AnnData object with topics as rows and peaks as columns,
    with binary values indicating whether a peak is present in a topic.

    Example
    -------
    >>> anndata = enhancerai.import_topics(
    ...     topics_folder="path/to/topics",
    ...     peaks_file="path/to/peaks.bed",
    ...     topics_subset=["Topic_1", "Topic_2"],
    ... )

    Parameters
    ----------
    topics_folder
        Folder name containing the topic BED files.
    peaks_file
        File name of the consensus regions BED file.
    topics_subset
        List of topics to include in the AnnData object. If None, all topics
        will be included.
        Topics should be named after the topics file name without the extension.
    compress
        Compress the AnnData.X matrix. If True, the matrix will be stored as
        a sparse matrix. If False, the matrix will be stored as a dense matrix.

    Returns
    -------
    AnnData object with topics as rows and peaks as columns.
    """
    topics_folder = Path(topics_folder)
    peaks_file = Path(peaks_file)

    # Input checks
    if not topics_folder.is_dir():
        raise FileNotFoundError(f"Directory '{topics_folder}' not found")
    if not peaks_file.is_file():
        raise FileNotFoundError(f"File '{peaks_file}' not found")
    if topics_subset is not None:
        for topic in topics_subset:
            if not any(topics_folder.glob(f"{topic}.bed")):
                raise FileNotFoundError(
                    f"Topic '{topic}' not found in '{topics_folder}'"
                )

    # Read consensus regions BED file
    consensus_peaks = pd.read_csv(peaks_file, sep="\t", header=None, usecols=[0, 1, 2])
    consensus_peaks["region"] = (
        consensus_peaks[0].astype(str)
        + ":"
        + consensus_peaks[1].astype(str)
        + "-"
        + consensus_peaks[2].astype(str)
    )

    binary_matrix = pd.DataFrame(0, index=[], columns=consensus_peaks["region"])
    topic_file_paths = []

    # Which topic regions are present in the consensu regions
    for topic_file in sorted(topics_folder.glob("*.bed")):
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
        warnings.warn(
            f"{len(regions_no_topics.index)} consensus regions are not open in any topic",
            stacklevel=1,
        )

    return ann_data
