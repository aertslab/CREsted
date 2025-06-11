from __future__ import annotations

import os
import re

import anndata
import h5py
import modiscolite as modisco
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger

from crested.utils._logging import log_and_raise

from ._modisco_utils import (
    _pattern_to_ppm,
    _trim_pattern_by_ic,
    compute_ic,
    match_score_patterns,
    read_html_to_dataframe,
    write_to_meme,
)


def _calculate_window_offsets(center: int, window_size: int) -> tuple:
    return (center - window_size // 2, center + window_size // 2)


@log_and_raise(Exception)
def tfmodisco(
    contrib_dir: os.PathLike = "modisco_results",
    class_names: list[str] | None = None,
    output_dir: os.PathLike = "modisco_results",
    max_seqlets: int = 5000,
    min_metacluster_size: int = 100,
    min_final_cluster_size: int = 20,
    window: int = 500,
    n_leiden: int = 2,
    report: bool = False,
    meme_db: str = None,
    verbose: bool = True,
    fdr: float = 0.05,
    sliding_window_size: int = 20,
    flank_size: int = 5,
    top_n_regions: int | None = None,
):
    """
    Run tf-modisco on one-hot encoded sequences and contribution scores stored in .npz files.

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
    min_metacluster_size
        Minimum number of seqlets per metacluster.
    min_final_cluster_size
        Minimum size of final cluster.
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
    fdr
        False discovery rate of seqlet finding.
    sliding_window_size
        Sliding window size for seqlet finding in tfmodiscolite.
    flank_size
        Flank size of seqlets.
    top_n_regions
        The top n regions from the one hot encoded sequences and contribution scores to run modisco on.

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
    >>> crested.tl.modisco.tfmodisco(
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

            if top_n_regions:
                top_n = (
                    top_n_regions if top_n_regions < len(sequences) else len(sequences)
                )
                top_n = max(top_n, 1)  # avoid faulty inputs
                sequences = sequences[:top_n]
                attributions = attributions[:top_n]

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
                    sliding_window_size=sliding_window_size,
                    flank_size=flank_size,
                    target_seqlet_fdr=fdr,
                    n_leiden_runs=n_leiden,
                    verbose=verbose,
                    min_metacluster_size=min_metacluster_size,
                    final_min_cluster_size=min_final_cluster_size,
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


def get_pwms_from_modisco_file(
    modisco_file: str,
    min_ic: float = 0.1,
    output_meme_file: str | None = None,
    metacluster_name: str | None = None,
    pattern_indices: list[int] | None = None,
):
    """
    Extract PPMs (Position Probability Matrices) from a Modisco HDF5 results file.

    Optionally, save the extracted PPMs in MEME format.

    Parameters
    ----------
    modisco_file : str
        Path to the Modisco HDF5 results file.
    min_ic : float
        Threshold to trim pattern. The higher, the more it gets trimmed.
    output_meme_file : str | None
        Path to save the extracted PPMs in MEME format. If None, PPMs are not saved.
    metacluster_name : str | None
        The name of the metacluster to process (e.g., 'pos_patterns' or 'neg_patterns').
        If None, all metaclusters are processed.
    pattern_indices : list[int] | None
        List of pattern indices to include from the selected metacluster.
        If None, all patterns are processed.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary where keys are pattern IDs (e.g., "pos_patterns_pattern_0")
        and values are numpy arrays of PPMs.
    """
    ppms = {}

    # Issue a warning if pattern_indices are provided without a metacluster_name
    if pattern_indices and not metacluster_name:
        logger.info(
            "Pattern indices are specified, but no metacluster name is provided. "
            "Pattern indices will be ignored."
        )
        pattern_indices = None

    # Open the HDF5 file
    with h5py.File(modisco_file, "r") as hdf5_results:
        # Select specific metacluster or iterate through all metaclusters
        metaclusters_to_process = (
            [metacluster_name] if metacluster_name else hdf5_results.keys()
        )

        for metacluster in metaclusters_to_process:
            if metacluster not in hdf5_results:
                raise ValueError(
                    f"Metacluster '{metacluster}' not found in the HDF5 file."
                )

            pos_pat = metacluster == "pos_patterns"
            metacluster_data = hdf5_results[metacluster]

            # Select specific patterns or iterate through all patterns
            patterns_to_process = (
                pattern_indices if pattern_indices else range(len(metacluster_data))
            )

            for i in patterns_to_process:
                pattern_name = f"pattern_{i}"
                if pattern_name not in metacluster_data:
                    raise ValueError(
                        f"Pattern '{pattern_name}' not found in metacluster '{metacluster}'."
                    )

                pattern = metacluster_data[pattern_name]
                pattern_trimmed = _trim_pattern_by_ic(pattern, pos_pat, min_ic)

                # Extract the PPM as a numpy array
                ppm = np.array(pattern_trimmed["sequence"])

                # Save the PPM with a unique key
                ppms[f"{metacluster}_{pattern_name}"] = ppm

    # Optionally write PPMs to a MEME file
    if output_meme_file:
        write_to_meme(ppms, output_meme_file)

    return ppms


def add_pattern_to_dict(
    p: dict[str, np.ndarray],
    idx: int,
    cell_type: str,
    pos_pattern: bool,
    all_patterns: dict,
) -> dict:
    """
    Add a pattern to the dictionary.

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
    ppm = _pattern_to_ppm(p)
    ic, ic_pos, ic_mat = compute_ic(ppm)

    p["ppm"] = ppm
    p["ic"] = np.mean(ic_pos)
    all_patterns[str(idx)] = {}
    all_patterns[str(idx)]["pattern"] = p
    all_patterns[str(idx)]["pos_pattern"] = pos_pattern

    all_patterns[str(idx)]["ppm"] = ppm
    all_patterns[str(idx)]["ic"] = np.mean(
        ic_pos
    )  # np.mean(_get_ic(p["contrib_scores"], pos_pattern))
    all_patterns[str(idx)]["instances"] = {}
    all_patterns[str(idx)]["instances"][p["id"]] = p
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
    Match the pattern to existing patterns and updates the dictionary.

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
    p["pos_pattern"] = pos_pattern
    p["n_seqlets"] = p["seqlets"]["n_seqlets"][0]
    if not all_patterns:
        return add_pattern_to_dict(p, 0, cell_type, pos_pattern, all_patterns)

    match = False
    match_idx = None
    max_sim = 0

    ppm = _pattern_to_ppm(p)
    ic, ic_pos, ic_mat = compute_ic(ppm)
    p_ic = np.mean(ic_pos)
    p["ic"] = p_ic
    p["ppm"] = ppm

    p["class"] = cell_type

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
            f'Match between {pattern_id} and {all_patterns[str(match_idx)]["pattern"]["id"]} with similarity score {max_sim:.2f}'
        )

    all_patterns[str(match_idx)]["instances"][pattern_id] = p

    if cell_type in all_patterns[str(match_idx)]["classes"].keys():
        ic_class_representative = all_patterns[str(match_idx)]["classes"][cell_type][
            "ic"
        ]
        n_seqlets_class_representative = all_patterns[str(match_idx)]["classes"][
            cell_type
        ]["n_seqlets"]
        if p_ic > ic_class_representative:
            all_patterns[str(match_idx)]["classes"][cell_type] = p
        all_patterns[str(match_idx)]["classes"][cell_type]["n_seqlets"] = (
            n_seqlets_class_representative + p["n_seqlets"]
        )  # We add to the total number of seqlets for this class
    else:
        all_patterns[str(match_idx)]["classes"][cell_type] = p

    if p_ic > all_patterns[str(match_idx)]["ic"]:
        all_patterns[str(match_idx)]["ic"] = p_ic
        all_patterns[str(match_idx)]["pattern"] = p
        all_patterns[str(match_idx)]["ppm"] = ppm

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

    def should_merge(p1, p2):
        """Check if two patterns should merge based on the similarity threshold."""
        sim = max(
            match_score_patterns(p1["pattern"], p2["pattern"]),
            match_score_patterns(p2["pattern"], p1["pattern"]),
        )
        return sim > sim_threshold, sim

    iterations = 0  # Track number of iterations for debugging
    while True:
        merged_patterns = {}
        new_index = 0
        merged_indices = set()
        any_merged = False

        # Keep track of which pairs have been compared and the result
        for i, (idx1, pattern1) in enumerate(pattern_list):
            if idx1 in merged_indices:
                continue
            merged_indices.add(idx1)
            merged_pattern = pattern1.copy()

            for j, (idx2, pattern2) in enumerate(pattern_list):
                if i >= j or idx2 in merged_indices:
                    continue

                should_merge_result, similarity = should_merge(pattern1, pattern2)

                if should_merge_result:
                    merged_indices.add(idx2)
                    merged_pattern = merge_patterns(merged_pattern, pattern2)
                    any_merged = True
                    if verbose:
                        print(
                            f'Merged patterns {pattern1["pattern"]["id"]} and {pattern2["pattern"]["id"]} with similarity {similarity}'
                        )
            # Add the merged pattern to the new set of patterns
            merged_patterns[str(new_index)] = merged_pattern
            new_index += 1

        # If nothing merged in this pass, break out
        if not any_merged:
            break

        iterations += 1  # Increment number of iterations
        if verbose:
            print(f"Iteration {iterations}: Merging complete, checking again")

        # Rebuild pattern list from merged patterns for the next iteration
        pattern_list = list(merged_patterns.items())

    # Final filtering based on IC discard threshold
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

    if verbose:
        print(f"Total iterations: {iterations}")

    return final_patterns


def merge_patterns(pattern1: dict, pattern2: dict) -> dict:
    """
    Merge two patterns into one. The resulting pattern will have the highest IC pattern as the representative pattern.

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
    merged_classes = {}
    for cell_type in pattern1["classes"].keys():
        if cell_type in pattern2["classes"].keys():
            ic_a = pattern1["classes"][cell_type]["ic"]
            n_seqlets_a = pattern1["classes"][cell_type]["n_seqlets"]
            ic_b = pattern2["classes"][cell_type]["ic"]
            n_seqlets_b = pattern2["classes"][cell_type]["n_seqlets"]
            merged_classes[cell_type] = (
                pattern1["classes"][cell_type]
                if ic_a > ic_b
                else pattern2["classes"][cell_type]
            )
            merged_classes[cell_type]["n_seqlets"] = n_seqlets_a + n_seqlets_b
        else:
            merged_classes[cell_type] = pattern1["classes"][cell_type]

    for cell_type in pattern2["classes"].keys():
        if cell_type not in merged_classes.keys():
            merged_classes[cell_type] = pattern2["classes"][cell_type]

    merged_instances = {**pattern1["instances"], **pattern2["instances"]}

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
        "instances": merged_instances,
    }


def pattern_similarity(all_patterns: dict, idx1: int, idx2: int) -> float:
    """
    Compute the similarity between two patterns.

    Parameters
    ----------
    all_patterns
        dictionary containing all patterns.
    idx1
        Index of the first pattern.
    idx2
        Index of the second pattern.

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


def find_pattern(pattern_id: str, pattern_dict: dict) -> int | None:
    """
    Find the index of a pattern by its ID.

    Parameters
    ----------
    pattern_id
        The ID of the pattern to find.
    pattern_dict
        A dictionary containing pattern data.

    Returns
    -------
    The index of the pattern if found, otherwise None.
    """
    for idx, p in enumerate(pattern_dict):
        if pattern_id == pattern_dict[p]["pattern"]["id"]:
            return idx
        for c in pattern_dict[p]["classes"]:
            if pattern_id == pattern_dict[p]["classes"][c]["id"]:
                return idx
    return None


def match_h5_files_to_classes(
    contribution_dir: str, classes: list[str]
) -> dict[str, str | None]:
    """
    Match .h5 files in a given directory with a list of class names and returns a dictionary mapping.

    Parameters
    ----------
    contribution_dir
        Directory containing .h5 files.
    classes
        list of class names to match against file names.

    See Also
    --------
    crested.tl.modisco.tfmodisco

    Returns
    -------
    A dictionary where keys are class names and values are paths to the corresponding .h5 files if matched, None otherwise.
    """
    h5_files = [file for file in os.listdir(contribution_dir) if file.endswith(".h5")]
    matched_files = dict.fromkeys(classes, None)

    for file in h5_files:
        base_name = os.path.splitext(file)[0][:-16]
        for class_name in classes:
            if base_name == class_name:
                matched_files[class_name] = os.path.join(contribution_dir, file)
                break

    return matched_files

def _read_and_trim_patterns(
    cell_type: str,
    file_list: str | list[str],
    trim_ic_threshold: float,
    verbose: bool
) -> tuple[list[dict], list[str], list[bool]]:
    """
    Read and trim patterns from HDF5 files for a specific cell type.

    This function iterates over all HDF5 files associated with a cell type, reads the patterns stored in each metacluster,
    trims them using an information content threshold, and collects associated metadata.

    Parameters
    ----------
    cell_type
        The name of the cell type whose patterns are being processed.
    file_list
        Path(s) to HDF5 file(s) containing patterns for the cell type.
    trim_ic_threshold
        Information content threshold for trimming each pattern.
    verbose
        Whether to print diagnostic information during reading and processing.

    Returns
    -------
    trimmed_patterns
        A list of trimmed pattern dictionaries, one per pattern found.
    pattern_ids
        A list of pattern identifiers, each uniquely naming a pattern.
    is_pattern_pos
        A list of boolean flags indicating whether the pattern came from the "pos_patterns" metacluster.
    """
    trimmed_patterns = []
    pattern_ids = []
    is_pattern_pos = []

    if isinstance(file_list, str):
        file_list = [file_list]

    for h5_file in file_list:
        if verbose:
            print(f"Reading file {h5_file}")
        try:
            with h5py.File(h5_file) as hdf5_results:
                for metacluster_name in list(hdf5_results.keys()):
                    pattern_idx = 0
                    for i in range(len(list(hdf5_results[metacluster_name]))):
                        p = "pattern_" + str(i)
                        pattern_ids.append(
                            f"{cell_type.replace(' ', '_')}_{metacluster_name}_{pattern_idx}"
                        )
                        is_pos = metacluster_name == "pos_patterns"
                        pattern = _trim_pattern_by_ic(
                            hdf5_results[metacluster_name][p],
                            is_pos,
                            trim_ic_threshold,
                        )
                        pattern["file_path"] = h5_file
                        trimmed_patterns.append(pattern)
                        is_pattern_pos.append(is_pos)
                        pattern_idx += 1
        except OSError:
            print(f"File error at {h5_file}")
            continue

    return trimmed_patterns, pattern_ids, is_pattern_pos

def calculate_tomtom_similarity_per_pattern(
    matched_files: dict[str, str | list[str] | None],
    trim_ic_threshold: float = 0.05,
    use_ppm: bool = False,
    background_freqs: list | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, dict]]:
    """
    Compute pairwise similarity between all trimmed patterns across matched HDF5 files using TOMTOM.

    This function reads in motif patterns from HDF5 files (e.g., from a TF-MoDISco pipeline),
    trims them based on information content, converts them to PPMs, and computes a full pairwise
    similarity matrix using TOMTOM. It also returns pattern metadata, including the contribution
    scores and the number of seqlets per pattern.

    Parameters
    ----------
    matched_files
        Dictionary mapping cell type names (or class names) to HDF5 file paths or list of paths
        containing TF-MoDISco results. A value of None indicates no data for that cell type.
    trim_ic_threshold
        Threshold for trimming low-information-content ends of patterns.
        Defaults to 0.05.
    verbose
        If True, prints progress messages.

    Returns
    -------
    similarity_matrix
        A 2D square NumPy array of shape (N, N), where N is the number of trimmed patterns across
        all cell types. Each entry [i, j] contains the TOMTOM similarity score (-log10 p-value)
        between pattern i and pattern j.
    all_pattern_ids
        A list of unique pattern identifiers, corresponding to the rows and columns in
        `similarity_matrix`.
    pattern_dict
        A dictionary mapping each pattern ID to a dictionary containing:
            - 'contrib_scores': the contribution score matrix (for visualization),
            - 'n_seqlets': the number of seqlets contributing to the pattern.

    Notes
    -----
    - Patterns are first trimmed using `_read_and_trim_patterns`.
    - PPMs are computed using `_pattern_to_ppm` and inserted into each pattern dictionary.
    - Similarity is computed using `match_score_patterns`, which uses TOMTOM under the hood.
    - The function assumes the presence of external dependencies like `_read_and_trim_patterns`,
      `_pattern_to_ppm`, and `match_score_patterns`, typically from a motif analysis library.
    """
    if background_freqs is None:
        background_freqs = [0.28, 0.22, 0.22, 0.28]
    background_freqs = np.array(background_freqs)

    all_trimmed_patterns = []
    all_pattern_ids = []

    for cell_type in matched_files:
        trimmed_patterns, pattern_ids, is_pattern_pos = _read_and_trim_patterns(
            cell_type,
            matched_files[cell_type],
            trim_ic_threshold,
            verbose
        )
        all_trimmed_patterns += trimmed_patterns
        all_pattern_ids += pattern_ids

    # Add PPMs to each pattern
    pattern_ppms = [_pattern_to_ppm(p) for p in all_trimmed_patterns]
    for i, pat in enumerate(all_trimmed_patterns):
        pat['ppm'] = pattern_ppms[i]

    if verbose:
        print('Total patterns:', len(all_trimmed_patterns))

    # Compute pairwise TOMTOM similarity
    similarity_matrix = match_score_patterns(all_trimmed_patterns, all_trimmed_patterns, use_ppm=use_ppm, background_freqs=background_freqs)

    # Construct output metadata dictionary
    pattern_dict = {
        pid: {
            'contrib_scores': all_trimmed_patterns[i]['contrib_scores'],
            'n_seqlets': all_trimmed_patterns[i]['seqlets']['n_seqlets']
        }
        for i, pid in enumerate(all_pattern_ids)
    }

    return similarity_matrix, all_pattern_ids, pattern_dict

def process_patterns(
    matched_files: dict[str, str | list[str] | None],
    sim_threshold: float = 3.0,
    trim_ic_threshold: float = 0.05,
    discard_ic_threshold: float = 0.1,
    verbose: bool = False,
) -> dict[str, dict[str, str | list[float]]]:
    """
    Process genomic patterns from matched HDF5 files, trim based on information content, and match to known patterns.

    Parameters
    ----------
    matched_files
        dictionary with class names as keys and paths to HDF5 files as values.
    sim_threshold
        Similarity threshold for matching patterns (-log10(pval), pval obtained through TOMTOM matching from memesuite-lite)
    trim_ic_threshold
        Information content threshold for trimming patterns.
    discard_ic_threshold
        Information content threshold for discarding patterns.
    verbose
        Flag to enable verbose output.

    See Also
    --------
    crested.tl.modisco.match_h5_files_to_classes

    Returns
    -------
    All processed patterns with metadata.
    """
    all_patterns = {}

    for cell_type in matched_files:

        trimmed_patterns, pattern_ids, is_pattern_pos = _read_and_trim_patterns(
            cell_type,
            matched_files[cell_type],
            trim_ic_threshold,
            verbose
        )

        for idx, p in enumerate(trimmed_patterns):
            all_patterns = match_to_patterns(
                p,
                idx,
                cell_type,
                pattern_ids[idx],
                is_pattern_pos[idx],
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
    pattern_parameter: str = "seqlet_count",
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
    pattern_parameter
        Parameter which is used to indicate the pattern's importance. Either average contribution score ('contrib'), or number of pattern instances ('seqlet_count', default) and its log ('seqlet_count_log').

    See Also
    --------
    crested.tl.modisco.process_patterns
    crested.pl.patterns.clustermap

    Returns
    -------
    The resulting pattern matrix, optionally normalized.
    """
    if pattern_parameter not in ["contrib", "seqlet_count", "seqlet_count_log"]:
        logger.info("Pattern parameter not valid. Setting to default ('seqlet_count')")
        pattern_parameter = "seqlet_count"

    pattern_matrix = np.zeros((len(classes), len(all_patterns.keys())))

    for p_idx in all_patterns:
        p_classes = list(all_patterns[p_idx]["classes"].keys())
        for ct in p_classes:
            idx = np.argwhere(np.array(classes) == ct)[0][0]
            avg_contrib = np.mean(all_patterns[p_idx]["classes"][ct]["contrib_scores"])
            if pattern_parameter == "contrib":
                pattern_matrix[idx, int(p_idx)] = avg_contrib
            elif pattern_parameter == "seqlet_count":
                sign = (
                    1 if avg_contrib > 0 else -1
                )  # Negative patterns will have a 'negative' count to reflect the negative performance.
                pattern_matrix[idx, int(p_idx)] = (
                    sign * all_patterns[p_idx]["classes"][ct]["n_seqlets"]
                )
            elif pattern_parameter == "seqlet_count_log":
                sign = (
                    1 if avg_contrib > 0 else -1
                )  # Negative patterns will have a 'negative' count to reflect the negative performance.
                pattern_matrix[idx, int(p_idx)] = sign * np.log1p(
                    all_patterns[p_idx]["classes"][ct]["n_seqlets"]
                )
            else:
                raise ValueError(
                    "Invalid pattern_parameter. Set to either 'contrib' or 'seqlet_count'."
                )

    # Filter out columns that are all zeros
    filtered_array = pattern_matrix[:, ~np.all(pattern_matrix == 0, axis=0)]

    if normalize:
        filtered_array = normalize_rows(filtered_array)

    return filtered_array


def calculate_similarity_matrix(all_patterns: dict) -> np.ndarray:
    """
    Calculate the similarity matrix for the given patterns.

    Parameters
    ----------
    all_patterns
        Dictionary containing pattern data. Each key is a pattern index, and each value is a dictionary with pattern information.

    Returns
    -------
    A 2D numpy array containing the similarity values.

    See Also
    --------
    crested.pl.patterns.similarity_heatmap
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

    See Also
    --------
    crested.tl.modisco.process_patterns

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

    for pat_id in all_patterns:
        html_paths_pattern = []
        for pattern in all_patterns[pat_id]["instances"]:
            pattern_id = all_patterns[pat_id]["instances"][pattern]["id"]
            pattern_class_parts = pattern_id.split("_")[:-3]
            pattern_class = (
                "_".join(pattern_class_parts)
                if len(pattern_class_parts) > 1
                else pattern_class_parts[0]
            )

            html_dir = os.path.join(contribution_dir, pattern_class + "_report")
            html_paths_pattern.append(os.path.join(html_dir, "motifs.html"))
        html_paths.append(html_paths_pattern)

    return html_paths


def find_pattern_matches(
    all_patterns: dict, html_paths: list[str], q_val_thr: float = 0.05
) -> dict[int, dict[str, list[str]]]:
    """
    Find and filter pattern matches from the modisco-lite list of patterns to the motif database from the corresponding HTML paths.

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
        matching_rows = []
        pattern_ids = []
        for j, pattern in enumerate(all_patterns[p_idx]["instances"]):
            df_motif_database = read_html_to_dataframe(html_paths[i][j])
            if not isinstance(df_motif_database, pd.DataFrame):
                logger.warning(
                    f"Skipping pattern match: expected DataFrame but got {type(df_motif_database).__name__}.\n"
                    f"Problematic HTML path: {html_paths[i][j]}"
                )
                continue
            pattern_id_whole = all_patterns[p_idx]["instances"][pattern]["id"]
            pattern_id_parts = pattern_id_whole.split("_")
            pattern_id = (
                pattern_id_parts[-3]
                + "_"
                + pattern_id_parts[-2]
                + "."
                + "pattern"
                + "_"
                + pattern_id_parts[-1]
            )
            matching_row = df_motif_database.loc[
                df_motif_database["pattern"] == pattern_id
            ]
            matching_rows.append(matching_row)
            pattern_ids.append(pattern_id_whole)

        # Process the matching rows if found
        if len(matching_rows) > 0:
            matches = []
            matches_filt = []

            patterns = []
            for i, matching_row in enumerate(matching_rows):
                if not matching_row.empty:
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

                    for match in matches:
                        if match.startswith("metacluster"):
                            match_parts = match.split(".")[2:]
                            match = ".".join(match_parts)
                        matches_filt.append(match)
                        patterns.append(pattern_ids[i])

            if matches_filt:
                pattern_match_dict[p_idx] = {
                    "matches": matches_filt,
                    "patterns": patterns,
                }
        else:
            print(f"No matching row found for pattern_id '{pattern_id}'")

    return pattern_match_dict


def read_motif_to_tf_file(file_path: str) -> pd.DataFrame:
    """
    Read a TSV file mapping motifs to transcription factors (TFs) into a DataFrame.

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
    Create a dictionary mapping patterns to their associated transcription factors (TFs) and other metadata.

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
    normalize_pattern_importances: bool = False,
    normalize_gex: bool = False,
    min_tf_gex: float = 0,
    importance_threshold: float = 0,
    pattern_parameter: str = "seqlet_count",
    filter_correlation: bool = False,
    zscore_threshold: float = 2,
    correlation_threshold: float = 0.2,
    verbose: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Create a tensor (matrix) of transcription factor (TF) expression and cell type contributions.

    Parameters
    ----------
    pattern_tf_dict
        A dictionary with pattern indices and their TFs. See `crested.tl.modisco.create_pattern_tf_dict`.
    all_patterns
        A list of patterns with metadata. See `crested.tl.modisco.process_patterns`.
    df
        A DataFrame containing gene expression data. See `crested.tl.modisco.calculate_mean_expression_per_cell_type`
    classes
        A list of cell type classes.
    log_transform
        Whether to apply log transformation to the gene expression values. Default is True.
    normalize_pattern_importances
        Whether to normalize the contribution scores across the cell types. Default is False.
    normalize_gex
        Whether to normalize gene expression across the cell types. Default is False.
    min_tf_gex
        The minimal GEX value to select potential TF candidates. Default 0.
    importance_threshold
        The minimum pattern importance value. Default is 0.
    pattern_parameter
        Parameter which is used to indicate the pattern's importance. Either average contribution score ('contrib'), or number of pattern instances ('seqlet_count', default) and its log ('seqlet_count_log').
    filter_correlation
        Whether to filter based on Pearson correlation between `tf_gex` and `ct_contribs`. Default is False.
    zscore_threshold
        Zscore used for filtering TF candidates. If the max zscore over the cell types is belofw this threshold, the TF gets discarded. Default is 2.
    correlation_threshold
        Minimum Pearson correlation between expression and contribution profile required to keep a column if filtering is enabled. Default is 0.2.
    verbose
        Whether to print intermediate debugging steps.

    See Also
    --------
    crested.tl.modisco.create_pattern_tf_dict, crested.tl.modisco.process_patterns, crested.tl.modisco.calculate_mean_expression_per_cell_type

    Returns
    -------
    A tuple containing the TF-cell type matrix and the list of TF pattern annotations.
    """
    total_tf_patterns = sum(len(pattern_tf_dict[p]["tfs"]) for p in pattern_tf_dict)
    tf_ct_matrix = np.zeros((len(classes), total_tf_patterns, 2))
    tf_pattern_annots = []

    df = df.reindex(classes) # Ensure they are in same order.

    if pattern_parameter not in ["contrib", "seqlet_count", "seqlet_count_log"]:
        logger.info("Pattern parameter not valid. Setting to default ('seqlet_count').")
        pattern_parameter = "seqlet_count"

    counter = 0
    for p_idx in pattern_tf_dict:
        ct_contribs = np.zeros(len(classes))
        for ct in all_patterns[p_idx]["classes"]:
            idx = np.argwhere(np.array(classes) == ct)[0][0]
            contribs = np.mean(all_patterns[p_idx]["classes"][ct]["contrib_scores"])
            if pattern_parameter == "contrib":
                ct_contribs[idx] = contribs
            elif pattern_parameter == "seqlet_count":
                sign = 1 if contribs > 0 else -1
                ct_contribs[idx] = (
                    sign * all_patterns[p_idx]["classes"][ct]["n_seqlets"]
                )
            elif pattern_parameter == "seqlet_count_log":
                sign = 1 if contribs > 0 else -1
                ct_contribs[idx] = sign * np.log1p(
                    all_patterns[p_idx]["classes"][ct]["n_seqlets"]
                )
            else:
                raise ValueError(
                    "Invalid pattern_parameter. Set to either 'contrib' or 'seqlet_count'."
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
    if normalize_pattern_importances:
        tf_ct_matrix[:, :, 1] = normalize_rows(tf_ct_matrix[:, :, 1])
    if normalize_gex:
        tf_ct_matrix[:, :, 0] = normalize_rows(tf_ct_matrix[:, :, 0].T).T

    # Logic to remove columns where tf_gex is not above the expression threshold for all ct_contribs above the contribution threshold.
    initial_columns = tf_ct_matrix.shape[1]
    columns_to_keep = []

    for col in range(initial_columns):
        tf_gex_col = tf_ct_matrix[:, col, 0]
        ct_contribs_col = tf_ct_matrix[:, col, 1]

        # Identify relevant ct_contribs
        relevant_contribs = ct_contribs_col > importance_threshold

        # Check if there are valid ct_contribs and tf_gex above the threshold
        if np.any(relevant_contribs) and np.any(
            tf_gex_col[relevant_contribs] > min_tf_gex
        ):
            columns_to_keep.append(col)

    # Convert columns_to_keep to a boolean mask
    columns_to_keep = np.array(columns_to_keep)

    # Filter the matrix and annotations based on the columns_to_keep
    final_columns = len(columns_to_keep)
    removed_columns = initial_columns - final_columns

    tf_ct_matrix = tf_ct_matrix[:, columns_to_keep, :]
    tf_pattern_annots = [
        annot for i, annot in enumerate(tf_pattern_annots) if i in columns_to_keep
    ]

    if verbose:
        print(f"Total columns before threshold filtering: {initial_columns}")
        print(f"Total columns after threshold filtering: {final_columns}")
        print(f"Total columns removed: {removed_columns}")

    # Filter out TF candidates for patterns that do not show correlation between their expression and importance profiles.
    if filter_correlation:
        initial_columns = tf_ct_matrix.shape[1]
        columns_to_keep = []

        for col in range(initial_columns):
            tf_gex_col = tf_ct_matrix[:, col, 0]
            ct_contribs_col = np.abs(tf_ct_matrix[:, col, 1])

            tf_gex_col_z = (tf_gex_col - np.mean(tf_gex_col)) / np.std(tf_gex_col)

            correlation = np.corrcoef(tf_gex_col, ct_contribs_col)[0, 1]
            # if correlation >= correlation_threshold:
            if (np.max(tf_gex_col_z) > zscore_threshold) and (
                correlation >= correlation_threshold
            ):
                columns_to_keep.append(col)

        # Update matrix and annotations based on filtering
        tf_ct_matrix = tf_ct_matrix[:, columns_to_keep, :]
        tf_pattern_annots = [
            annot for i, annot in enumerate(tf_pattern_annots) if i in columns_to_keep
        ]

        final_columns = len(columns_to_keep)
        removed_columns = initial_columns - final_columns

        if verbose:
            print(f"Total columns before correlation filtering: {initial_columns}")
            print(f"Total columns after correlation filtering: {final_columns}")
            print(f"Total columns removed: {removed_columns}")

    return tf_ct_matrix, tf_pattern_annots


def calculate_mean_expression_per_cell_type(
    file_path: str,
    cell_type_column: str,
    cpm_normalize: bool = False,
) -> pd.DataFrame:
    """
    Read an AnnData object from an H5AD file and calculates the mean gene expression per cell type subclass.

    Parameters
    ----------
    file_path
        The path to the H5AD file containing the single-cell RNA-seq data.
    cell_type_column
        The column name in the cell metadata that defines the cell type subclass.
    cpm_normalize
        Whether to additionally cpm_normalize the scRNA-seq data.

    Returns
    -------
    A DataFrame containing the mean gene expression per cell type subclass.
    """
    # Read the AnnData object from the specified H5AD file
    adata: anndata.AnnData = anndata.read_h5ad(file_path)

    # CPM normalize the counts if necessary
    if cpm_normalize:
        sc.pp.normalize_total(adata)

    # Convert the AnnData object to a DataFrame containing the gene expression matrix
    gene_expression_df: pd.DataFrame = adata.to_df()

    # Retrieve the cell metadata from the AnnData object
    cell_metadata: pd.DataFrame = adata.obs

    # Check if the specified cell type column exists in the cell metadata
    if cell_type_column not in cell_metadata.columns:
        raise ValueError(f"Column '{cell_type_column}' not found in cell metadata")

    # Calculate the mean gene expression per cell type subclass
    mean_expression_per_cell_type: pd.DataFrame = gene_expression_df.groupby(
        cell_metadata[cell_type_column]
    ).mean()

    return mean_expression_per_cell_type
