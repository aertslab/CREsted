from __future__ import annotations

import os
import re

import h5py
import modiscolite as modisco
import numpy as np
import pandas as pd
from loguru import logger

from crested._logging import log_and_raise
from crested.pl.patterns._modisco_results import (
    _get_ic,
    _trim_pattern_by_ic,
    plot_patterns,
)

from ._modisco_utils import match_score_patterns, read_html_to_dataframe


def _calculate_window_offsets(center: int, window_size: int) -> tuple:
    return (center - window_size // 2, center + window_size // 2)


@log_and_raise(Exception)
def tfmodisco(
    contrib_dir: os.PathLike = "modisco_results",
    class_names: list[str] | None = None,
    output_dir: os.PathLike = "modisco_results",
    max_seqlets: int = 5000,
    window: int = 500,
    n_leiden: int = 2,
    report: bool = False,
    meme_db: str = None,
    verbose: bool = True,
):
    """
    Runs tf-modisco on one-hot encoded sequences and contribution scores stored in .npz files.

    Parameters
    ----------
    contrib_dir
        Directory containing the contribution score and one hot encoded regions npz files.
    class_names
        list of class names to process. If None, all class names found in the output directory will be processed.
    output_dir
        Directory where output files will be saved.
    max_seqlets
        Maximum number of seqlets per metacluster.
    window
        The window surrounding the peak center that will be considered for motif discovery.
    n_leiden
        Number of Leiden clusterings to perform with different random seeds.
    report
        Generate a modisco report.
    meme_db
        Path to a MEME file (.meme) containing motifs. Required if report is True.
    verbose
        Print verbose output.

    See Also
    --------
    crested.tl.Crested.calculate_contribution_scores

    Examples
    --------
    >>> evaluator = crested.tl.Crested(...)
    >>> evaluator.load_model(/path/to/trained/model.keras)
    >>> evaluator.tfmodisco_calculate_and_save_contribution_scores(
    ...     adata, class_names=["Astro", "Vip"], method="expected_integrated_grad"
    ... )
    >>> crested.tl.tfmodisco(
    ...     contrib_dir="modisco_results",
    ...     class_names=["Astro", "Vip"],
    ...     output_dir="modisco_results",
    ...     window=1000,
    ... )
    """
    """Code adapted from https://github.com/jmschrei/tfmodisco-lite/blob/main/modisco."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if .npz exist in the contribution directory
    if not os.path.exists(contrib_dir):
        raise FileNotFoundError(f"Contribution directory not found: {contrib_dir}")
    files = os.listdir(contrib_dir)
    if not any(f.endswith(".npz") for f in files):
        raise FileNotFoundError("No .npz files found in the contribution directory")

    # Use all class names found in the contribution directory if class_names is not provided
    if class_names is None:
        class_names = [
            re.match(r"(.+?)_oh\.npz$", f).group(1)
            for f in os.listdir(contrib_dir)
            if f.endswith("_oh.npz")
        ]
        class_names = list(set(class_names))
        logger.info(
            f"No class names provided, using all found in the contribution directory: {class_names}"
        )

    # Iterate over each class and calculate contribution scores
    for class_name in class_names:
        try:
            # Load the one-hot sequences and contribution scores from the .npz files
            one_hot_path = os.path.join(contrib_dir, f"{class_name}_oh.npz")
            contrib_path = os.path.join(contrib_dir, f"{class_name}_contrib.npz")

            if not (os.path.exists(one_hot_path) and os.path.exists(contrib_path)):
                raise FileNotFoundError(f"Missing .npz files for class: {class_name}")

            one_hot_seqs = np.load(one_hot_path)["arr_0"]
            contribution_scores = np.load(contrib_path)["arr_0"]

            if one_hot_seqs.shape[2] < window:
                print(one_hot_seqs.shape[2])
                raise ValueError(
                    f"Window ({window}) cannot be longer than the sequences ({one_hot_seqs.shape[1]})"
                )

            center = one_hot_seqs.shape[2] // 2
            start, end = _calculate_window_offsets(center, window)

            sequences = one_hot_seqs[:, :, start:end]
            attributions = contribution_scores[:, :, start:end]

            sequences = sequences.transpose(0, 2, 1)
            attributions = attributions.transpose(0, 2, 1)

            sequences = sequences.astype("float32")
            attributions = attributions.astype("float32")

            # Define filenames for the output files
            output_file = os.path.join(output_dir, f"{class_name}_modisco_results.h5")
            report_dir = os.path.join(output_dir, f"{class_name}_report")

            # Check if the modisco results .h5 file does not exist for the class
            if not os.path.exists(output_file):
                logger.info(f"Running modisco for class: {class_name}")
                pos_patterns, neg_patterns = modisco.tfmodisco.TFMoDISco(
                    hypothetical_contribs=attributions,
                    one_hot=sequences,
                    max_seqlets_per_metacluster=max_seqlets,
                    sliding_window_size=20,
                    flank_size=5,
                    target_seqlet_fdr=0.05,
                    n_leiden_runs=n_leiden,
                    verbose=verbose,
                )

                modisco.io.save_hdf5(
                    output_file, pos_patterns, neg_patterns, window_size=window
                )

            else:
                print(f"Modisco results already exist for class: {class_name}")

            # Generate the modisco report
            if report and not os.path.exists(report_dir):
                modisco.report.report_motifs(
                    output_file,
                    report_dir,
                    meme_motif_db=meme_db,
                    top_n_matches=3,
                    is_writing_tomtom_matrix=False,
                    img_path_suffix="./",
                )

        except KeyError as e:
            logger.error(f"Missing data for class: {class_name}, error: {e}")


def add_pattern_to_dict(
    p: dict[str, np.ndarray],
    idx: int,
    cell_type: str,
    pos_pattern: bool,
    all_patterns: dict,
) -> dict:
    """
    Adds a pattern to the dictionary.

    Parameters
    ----------
    p
        Pattern to add.
    idx
        Index for the new pattern.
    cell_type
        Cell type of the pattern.
    pos_pattern
        Indicates if the pattern is a positive pattern.
    all_patterns
        dictionary containing all patterns.

    Returns
    -------
    Updated dictionary with the new pattern.
    """
    all_patterns[str(idx)] = {}
    all_patterns[str(idx)]["pattern"] = p
    all_patterns[str(idx)]["ic"] = np.mean(_get_ic(p["contrib_scores"], pos_pattern))
    all_patterns[str(idx)]["classes"] = {}
    all_patterns[str(idx)]["classes"][cell_type] = p
    return all_patterns


def match_to_patterns(
    p: dict,
    idx: int,
    cell_type: str,
    pattern_id: str,
    pos_pattern: bool,
    all_patterns: dict[str, dict[str, str | list[float]]],
    sim_threshold: float = 0.5,
    ic_threshold: float = 0.15,
    verbose: bool = False,
) -> dict:
    """
    Matches the pattern to existing patterns and updates the dictionary.

    Parameters
    ----------
    p
        Pattern to match.
    idx
        Index of the pattern.
    cell_type
        Cell type of the pattern.
    pattern_id
        ID of the pattern.
    pos_pattern
        Indicates if the pattern is a positive pattern.
    all_patterns
        dictionary containing all patterns.
    sim_threshold
        Similarity threshold for matching patterns.
    ic_threshold
        Information content threshold for matching patterns.
    verbose
        Flag to enable verbose output.

    Returns
    -------
    Updated dictionary with matched patterns.
    """
    p["id"] = pattern_id
    if not all_patterns:
        return add_pattern_to_dict(p, 0, cell_type, pos_pattern, all_patterns)

    match = False
    match_idx = None
    max_sim = 0

    for pat_idx, pattern in enumerate(all_patterns.keys()):
        sim = match_score_patterns(p, all_patterns[pattern]["pattern"])
        if sim > sim_threshold:
            match = True
            if sim > max_sim:
                max_sim = sim
                match_idx = pat_idx

    if not match:
        pattern_idx = len(all_patterns.keys())
        return add_pattern_to_dict(p, pattern_idx, cell_type, pos_pattern, all_patterns)

    if verbose:
        print(
            f'Match between {pattern_id} and {all_patterns[str(match_idx)]["pattern"]["id"]}'
        )
    all_patterns[str(match_idx)]["classes"][cell_type] = p
    p_ic = np.mean(_get_ic(p["contrib_scores"], pos_pattern))
    if p_ic > all_patterns[str(match_idx)]["ic"]:
        all_patterns[str(match_idx)]["ic"] = p_ic
        all_patterns[str(match_idx)]["pattern"] = p

    return all_patterns


def post_hoc_merging(
    all_patterns: dict,
    sim_threshold: float = 0.5,
    ic_discard_threshold: float = 0.15,
    verbose: bool = False,
) -> dict:
    """
    Double-checks the similarity of all patterns and merges them if they exceed the threshold.

    Filters out patterns with IC below the discard threshold at the last step and updates the keys.

    Parameters
    ----------
    all_patterns
        dictionary of all patterns with metadata.
    sim_threshold
        Similarity threshold for merging patterns.
    ic_discard_threshold
        IC threshold below which patterns are discarded.
    verbose
        Flag to enable verbose output of merged patterns.

    Returns
    -------
    Updated patterns after merging and filtering with sequential keys.
    """
    pattern_list = list(all_patterns.items())

    while True:
        merged_patterns = {}
        new_index = 0
        merged_indices = set()
        any_merged = False

        for i, (idx1, pattern1) in enumerate(pattern_list):
            if idx1 in merged_indices:
                continue
            merged_indices.add(idx1)
            merged_pattern = pattern1.copy()

            for j, (idx2, pattern2) in enumerate(pattern_list):
                if i >= j or idx2 in merged_indices:
                    continue

                sim = match_score_patterns(pattern1["pattern"], pattern2["pattern"])
                if sim > sim_threshold:
                    merged_indices.add(idx2)
                    merged_pattern = merge_patterns(merged_pattern, pattern2)
                    any_merged = True
                    if verbose:
                        print(
                            f'Merging patterns: {pattern1["pattern"]["id"]} and {pattern2["pattern"]["id"]} with similarity {sim}'
                        )

            merged_patterns[str(new_index)] = merged_pattern
            new_index += 1

        if not any_merged:
            break

        pattern_list = list(merged_patterns.items())

    # Final filtering based on IC discard threshold and class count
    filtered_patterns = {}
    discarded_ids = []

    for k, v in merged_patterns.items():
        if v["ic"] >= ic_discard_threshold or len(v["classes"]) > 1:
            filtered_patterns[k] = v
        else:
            discarded_ids.append(v["pattern"]["id"])

    if verbose:
        discarded_count = len(merged_patterns) - len(filtered_patterns)
        print(
            f"Discarded {discarded_count} patterns below IC threshold {ic_discard_threshold} and with a single class instance:"
        )
        print(discarded_ids)

    # Reindex the filtered patterns
    final_patterns = {
        str(new_idx): v for new_idx, (k, v) in enumerate(filtered_patterns.items())
    }

    return final_patterns


def merge_patterns(pattern1: dict, pattern2: dict) -> dict:
    """
    Merges two patterns into one. The resulting pattern will have the highest IC pattern as the representative pattern.

    Parameters
    ----------
    pattern1
        First pattern with metadata.
    pattern2
        Second pattern with metadata.

    Returns
    -------
    Merged pattern with updated metadata.
    """
    merged_classes = {**pattern1["classes"], **pattern2["classes"]}

    if pattern2["ic"] > pattern1["ic"]:
        representative_pattern = pattern2["pattern"]
        highest_ic = pattern2["ic"]
    else:
        representative_pattern = pattern1["pattern"]
        highest_ic = pattern1["ic"]

    return {
        "pattern": representative_pattern,
        "ic": highest_ic,
        "classes": merged_classes,
    }


def pattern_similarity(
    all_patterns: dict, idx1: int, idx2: int, plot: bool = False
) -> float:
    """
    Computes the similarity between two patterns.

    Parameters
    ----------
    all_patterns
        dictionary containing all patterns.
    idx1
        Index of the first pattern.
    idx2
        Index of the second pattern.
    plot
        Whether to plot the patterns.

    Returns
    -------
    Similarity score between the two patterns.
    """
    sim = max(
        match_score_patterns(
            all_patterns[str(idx1)]["pattern"], all_patterns[str(idx2)]["pattern"]
        ),
        match_score_patterns(
            all_patterns[str(idx2)]["pattern"], all_patterns[str(idx1)]["pattern"]
        ),
    )
    if plot:
        plot_patterns(all_patterns, [idx1, idx2])
    return sim


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    """
    Normalize the rows of an array such that the positive values are scaled by their maximum and negative values by their minimum absolute value.

    Parameters
    ----------
    arr
        Input array to be normalized.

    Returns
    -------
    The row-normalized array.
    """
    normalized_array = np.zeros_like(arr)

    for i in range(arr.shape[0]):
        pos_values = arr[i, arr[i] > 0]
        neg_values = arr[i, arr[i] < 0]

        if pos_values.size > 0:
            max_pos = np.max(pos_values)
            normalized_array[i, arr[i] > 0] = pos_values / max_pos

        if neg_values.size > 0:
            min_neg = np.min(neg_values)
            normalized_array[i, arr[i] < 0] = neg_values / abs(min_neg)

    return normalized_array


def find_pattern(id_: str, pattern_dict: dict) -> int | None:
    """
    Finds the index of a pattern by its ID.

    Parameters
    ----------
    id_
        The ID of the pattern to find.
    pattern_dict
        A dictionary containing pattern data.

    Returns
    -------
    The index of the pattern if found, otherwise None.
    """
    for idx, p in enumerate(pattern_dict):
        if id_ == pattern_dict[p]["pattern"]["id"]:
            return idx
        for c in pattern_dict[p]["classes"]:
            if id_ == pattern_dict[p]["classes"][c]["id"]:
                return idx
    return None


def match_h5_files_to_classes(
    contribution_dir: str, classes: list[str]
) -> dict[str, str | None]:
    """
    Matches .h5 files in a given directory with a list of class names and returns a dictionary mapping.

    Parameters
    ----------
    contribution_dir
        Directory containing .h5 files.
    classes
        list of class names to match against file names.

    Returns
    -------
    A dictionary where keys are class names and values are paths to the corresponding .h5 files if matched, None otherwise.
    """
    h5_files = [file for file in os.listdir(contribution_dir) if file.endswith(".h5")]
    matched_files = {class_name: None for class_name in classes}

    for file in h5_files:
        base_name = os.path.splitext(file)[0][:-16]
        for class_name in classes:
            if base_name == class_name:
                matched_files[class_name] = os.path.join(contribution_dir, file)
                break

    return matched_files


def process_patterns(
    matched_files: dict[str, str | None],
    sim_threshold: float = 0.5,
    trim_ic_threshold: float = 0.1,
    discard_ic_threshold: float = 0.15,
    verbose: bool = False,
) -> dict[str, dict[str, str | list[float]]]:
    """
    Process genomic patterns from matched HDF5 files, trim based on information content, and match to known patterns.

    Parameters
    ----------
    matched_files
        dictionary with class names as keys and paths to HDF5 files as values.
    sim_threshold
        Similarity threshold for matching patterns.
    trim_ic_threshold
        Information content threshold for trimming patterns.
    discard_ic_threshold
        Information content threshold for discarding patterns.
    verbose
        Flag to enable verbose output.

    Returns
    -------
    All processed patterns with metadata.
    """
    all_patterns = {}

    for _, cell_type in enumerate(matched_files.keys()):
        if verbose:
            print(f"Reading file {matched_files[cell_type]}...")
        if matched_files[cell_type] is None:
            continue
        try:
            hdf5_results = h5py.File(matched_files[cell_type], "r")
        except OSError:
            print(f"File error at {matched_files[cell_type]}")
            continue

        metacluster_names = list(hdf5_results.keys())
        patterns = []
        pattern_ids = []
        pos_patterns = []

        for metacluster_name in metacluster_names:
            for p in hdf5_results[metacluster_name]:
                pattern_ids.append(
                    f"{cell_type.replace(' ', '_')}_{metacluster_name}_{p}"
                )
                patterns.append(hdf5_results[metacluster_name][p])
                pos_pat = metacluster_name == "pos_patterns"
                pos_patterns.append(pos_pat)

        trimmed_patterns = [
            _trim_pattern_by_ic(pattern, pos_pattern, trim_ic_threshold)
            for pattern, pos_pattern in zip(patterns, pos_patterns)
        ]

        for idx, p in enumerate(trimmed_patterns):
            all_patterns = match_to_patterns(
                p,
                idx,
                cell_type,
                pattern_ids[idx],
                pos_patterns[idx],
                all_patterns,
                sim_threshold,
                discard_ic_threshold,
                verbose,
            )

    all_patterns = post_hoc_merging(
        all_patterns=all_patterns,
        sim_threshold=sim_threshold,
        ic_discard_threshold=discard_ic_threshold,
        verbose=verbose,
    )

    return all_patterns


def create_pattern_matrix(
    classes: list[str],
    all_patterns: dict[str, dict[str, str | list[float]]],
    normalize: bool = False,
) -> np.ndarray:
    """
    Create a pattern matrix from classes and patterns, with optional normalization.

    Parameters
    ----------
    classes
        list of class labels.
    all_patterns
        dictionary containing pattern data.
    normalize
        Flag to indicate whether to normalize the rows of the matrix.

    Returns
    -------
    The resulting pattern matrix, optionally normalized.
    """
    pattern_matrix = np.zeros((len(classes), len(all_patterns.keys())))

    for p_idx in all_patterns:
        p_classes = list(all_patterns[p_idx]["classes"].keys())
        for ct in p_classes:
            idx = np.argwhere(np.array(classes) == ct)[0][0]
            pattern_matrix[idx, int(p_idx)] = np.mean(
                all_patterns[p_idx]["classes"][ct]["contrib_scores"]
            )

    # Filter out columns that are all zeros
    filtered_array = pattern_matrix[:, ~np.all(pattern_matrix == 0, axis=0)]

    if normalize:
        filtered_array = normalize_rows(filtered_array)

    return filtered_array


def calculate_similarity_matrix(all_patterns: dict) -> np.ndarray:
    """
    Calculates the similarity matrix for the given patterns.

    Parameters
    ----------
    all_patterns
        Dictionary containing pattern data. Each key is a pattern index, and each value is a dictionary with pattern information.

    Returns
    -------
    A 2D numpy array containing the similarity values.
    """
    indices = list(all_patterns.keys())
    num_patterns = len(indices)

    similarity_matrix = np.zeros((num_patterns, num_patterns))

    for i, idx1 in enumerate(indices):
        for j, idx2 in enumerate(indices):
            similarity_matrix[i, j] = pattern_similarity(all_patterns, idx1, idx2)

    return similarity_matrix, indices


def generate_nucleotide_sequences(all_patterns: dict) -> list[tuple[str, np.ndarray]]:
    """
    Generate nucleotide sequences from pattern data.

    Parameters
    ----------
    all_patterns
        dictionary containing pattern data.

    Returns
    -------
    list of tuples containing sequences and their normalized heights.
    """
    nucleotide_map = {0: "A", 1: "C", 2: "G", 3: "T"}
    pat_seqs = []

    for p in all_patterns:
        c = np.abs(all_patterns[p]["pattern"]["contrib_scores"])
        max_indices = np.argmax(c, axis=1)
        max_values = np.max(c, axis=1)
        max_height = np.max(max_values)
        normalized_heights = max_values / max_height if max_height != 0 else max_values
        sequence = "".join([nucleotide_map[idx] for idx in max_indices])
        prefix = p + ":"
        sequence = prefix + sequence
        normalized_heights = np.concatenate((np.ones(len(prefix)), normalized_heights))
        pat_seqs.append((sequence, normalized_heights))

    return pat_seqs


def generate_image_paths(
    pattern_matrix: np.ndarray,
    all_patterns: dict,
    classes: list[str],
    contribution_dir: str,
) -> list[str]:
    """
    Generate image paths for each pattern in the filtered array.

    Parameters
    ----------
    pattern_matrix
        Filtered 2D array of pattern data.
    all_patterns
        Dictionary containing pattern data.
    classes
        List of class labels.
    contribution_dir
        Directory containing contribution scores and images.

    Returns
    -------
    List of image paths corresponding to the patterns.
    """
    image_paths = []

    for i in range(pattern_matrix.shape[1]):
        pattern_id = all_patterns[str(i)]["pattern"]["id"]
        pattern_class_parts = pattern_id.split("_")[:-4]
        pattern_class = (
            "_".join(pattern_class_parts)
            if len(pattern_class_parts) > 1
            else pattern_class_parts[0]
        )

        id_split = pattern_id.split("_")
        pos_neg = "pos_patterns." if id_split[-4] == "pos" else "neg_patterns."
        im_dir = contribution_dir
        im_path = f"{im_dir}{pattern_class}_report/trimmed_logos/{pos_neg}pattern_{id_split[-1]}.cwm.fwd.png"
        image_paths.append(im_path)

    return image_paths


def generate_html_paths(
    all_patterns: dict, classes: list[str], contribution_dir: str
) -> list[str]:
    """
    Generate html paths for each pattern in the filtered array.

    Parameters
    ----------
    pattern_matrix
        Filtered 2D array of pattern data.
    all_patterns
        dictionary containing pattern data.
    classes
        list of class labels.
    contribution_dir
        Directory containing contribution scores and images.

    Returns
    -------
    List of image paths corresponding to the patterns.
    """
    html_paths = []

    for i, _ in enumerate(all_patterns):
        pattern_id = all_patterns[str(i)]["pattern"]["id"]
        pattern_class_parts = pattern_id.split("_")[:-4]
        pattern_class = (
            "_".join(pattern_class_parts)
            if len(pattern_class_parts) > 1
            else pattern_class_parts[0]
        )

        html_dir = os.path.join(contribution_dir, pattern_class + "_report")
        html_paths.append(os.path.join(html_dir, "motifs.html"))

    return html_paths


def find_pattern_matches(
    all_patterns: dict, html_paths: list[str], q_val_thr: float = 0.05
) -> dict[int, dict[str, list[str]]]:
    """
    Finds and filters pattern matches from the modisco-lite list of patterns to the motif database from the corresponding HTML paths.

    Parameters
    ----------
    all_patterns
        A dictionary of patterns with metadata.
    html_paths
        A list of file paths to HTML files containing motif databases.
    q_val_thr
        The threshold for q-value filtering. Default is 0.05.

    Returns
    -------
    A dictionary with pattern indices as keys and a dictionary of matches as values.
    """
    pattern_match_dict: dict[int, dict[str, list[str]]] = {}

    for i, p_idx in enumerate(all_patterns):
        df_motif_database = read_html_to_dataframe(html_paths[i])
        pattern_id = all_patterns[p_idx]["pattern"]["id"]
        pattern_id_parts = pattern_id.split("_")
        pattern_id = (
            pattern_id_parts[-4]
            + "_"
            + pattern_id_parts[-3]
            + "."
            + pattern_id_parts[-2]
            + "_"
            + pattern_id_parts[-1]
        )
        matching_row = df_motif_database.loc[df_motif_database["pattern"] == pattern_id]

        # Process the matching row if found
        if not matching_row.empty:
            matches = []
            for j in range(3):
                qval_column = f"qval{j}"
                match_column = f"match{j}"
                if (
                    qval_column in matching_row.columns
                    and match_column in matching_row.columns
                ):
                    qval = matching_row[qval_column].values[0]
                    if qval < q_val_thr:
                        match = matching_row[match_column].values[0]
                        matches.append(match)

            matches_filt = []
            for match in matches:
                if match.startswith("metacluster"):
                    match_parts = match.split(".")[2:]
                    match = ".".join(match_parts)
                matches_filt.append(match)

            if matches_filt:
                pattern_match_dict[p_idx] = {"matches": matches_filt}
        else:
            print(f"No matching row found for pattern_id '{pattern_id}'")

    return pattern_match_dict


def read_motif_to_tf_file(file_path: str) -> pd.DataFrame:
    """
    Reads a TSV file mapping motifs to transcription factors (TFs) into a DataFrame.

    Parameters
    ----------
    file_path
        The path to the TSV file containing motif to TF mappings.

    Returns
    -------
    A DataFrame containing the motif to TF mappings.
    """
    return pd.read_csv(file_path, sep="\t")


def create_pattern_tf_dict(
    pattern_match_dict: dict,
    motif_to_tf_df: pd.DataFrame,
    all_patterns: dict,
    cols: list[str],
) -> tuple[dict, np.ndarray]:
    """
    Creates a dictionary mapping patterns to their associated transcription factors (TFs) and other metadata.

    Parameters
    ----------
    pattern_match_dict
        A dictionary with pattern indices and their matches.
    motif_to_tf_df
        A DataFrame containing motif to TF mappings.
    all_patterns
        A list of patterns with metadata.
    cols
        A list of column names to extract TF annotations from.

    Returns
    -------
    A tuple containing the pattern to TF mappings dictionary and an array of all unique TFs.
    """
    pattern_tf_dict: dict[int, dict[str]] = {}
    all_tfs: list[str] = []

    for _, p_idx in enumerate(pattern_match_dict):
        matches = pattern_match_dict[p_idx]["matches"]
        if len(matches) == 0:
            continue

        tf_list: list[list[str]] = []
        for match in matches:
            matching_row = motif_to_tf_df.loc[motif_to_tf_df["Motif_name"] == match]
            if not matching_row.empty:
                for col in cols:
                    annot = matching_row[col].values[0]
                    if not pd.isna(annot):
                        annot_list = annot.split(", ")
                        tf_list.append(annot_list)
                        all_tfs.append(annot_list)

        if len(tf_list) > 0:
            # Flatten the list of lists and get unique TFs
            tf_list_flat = [item for sublist in tf_list for item in sublist]
            unique_tfs = np.unique(tf_list_flat)

            pattern_tf_dict[p_idx] = {
                "pattern_info": all_patterns[p_idx],
                "tfs": unique_tfs,
                "matches": matches,
            }

    # Flatten all_tfs list and get unique TFs
    all_tfs_flat = [item for sublist in all_tfs for item in sublist]
    unique_all_tfs = np.sort(np.unique(all_tfs_flat))

    return pattern_tf_dict, unique_all_tfs


def create_tf_ct_matrix(
    pattern_tf_dict: dict,
    all_patterns: dict,
    df: pd.DataFrame,
    classes: list[str],
    log_transform: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Creates a tensor (matrix) of transcription factor (TF) expression and cell type contributions.

    Parameters
    ----------
    pattern_tf_dict
        A dictionary with pattern indices and their TFs.
    all_patterns
        A list of patterns with metadata.
    df
        A DataFrame containing gene expression data.
    classes
        A list of cell type classes.
    log_transform
        Whether to apply log transformation to the gene expression values. Default is True.
    normalize
        Whether to normalize the contribution scores across the cell types. Default is True.

    Returns
    -------
    A tuple containing the TF-cell type matrix and the list of TF pattern annotations.
    """
    total_tf_patterns = sum(len(pattern_tf_dict[p]["tfs"]) for p in pattern_tf_dict)
    tf_ct_matrix = np.zeros((len(classes), total_tf_patterns, 2))
    tf_pattern_annots = []

    counter = 0
    for p_idx in pattern_tf_dict:
        ct_contribs = np.zeros(len(classes))
        for ct in all_patterns[p_idx]["classes"]:
            idx = np.argwhere(np.array(classes) == ct)[0][0]
            ct_contribs[idx] = np.mean(
                all_patterns[p_idx]["classes"][ct]["contrib_scores"]
            )

        for tf in pattern_tf_dict[p_idx]["tfs"]:
            if tf in df.columns:
                tf_gex = df[tf].values
                if log_transform:
                    tf_gex = np.log(tf_gex + 1)

                tf_ct_matrix[:, counter, 0] = tf_gex
                tf_ct_matrix[:, counter, 1] = ct_contribs
                counter += 1
                tf_pattern_annot = tf + "_pattern_" + str(p_idx)
                tf_pattern_annots.append(tf_pattern_annot)

    tf_ct_matrix = tf_ct_matrix[:, : len(tf_pattern_annots), :]
    if normalize:
        tf_ct_matrix[:, :, 1] = normalize_rows(tf_ct_matrix[:, :, 1])

    return tf_ct_matrix, tf_pattern_annots
