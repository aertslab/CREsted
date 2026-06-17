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


def _run_modisco_single_class(
    class_name: str,
    *,
    contrib_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    window: int,
    max_seqlets: int,
    min_metacluster_size: int,
    min_final_cluster_size: int,
    n_leiden: int,
    report: bool,
    meme_db: str | None,
    verbose: bool,
    fdr: float,
    sliding_window_size: int,
    flank_size: int,
    top_n_regions: int | None,
    pin_numba_threads: bool = False,
) -> None:
    """Run tf-modisco for a single class. See `tfmodisco` for parameter semantics.

    This is the body of the per-class loop, factored out so it can be dispatched
    either serially or across processes (see the `n_jobs` argument of `tfmodisco`).
    """
    # When running across worker processes, pin numba to a single thread so that
    # N concurrent workers don't each spawn a full numba threadpool and
    # oversubscribe the cores (BLAS/OpenMP are pinned via joblib's
    # inner_max_num_threads in the caller).
    if pin_numba_threads:
        try:
            import numba

            numba.set_num_threads(1)
        except Exception:  # noqa: BLE001 - thread pinning is best-effort
            pass

    try:
        # Load the one-hot sequences and contribution scores from the .npz files
        one_hot_path = os.path.join(contrib_dir, f"{class_name}_oh.npz")
        contrib_path = os.path.join(contrib_dir, f"{class_name}_contrib.npz")

        if not (os.path.exists(one_hot_path) and os.path.exists(contrib_path)):
            raise FileNotFoundError(f"Missing .npz files for class: {class_name}")

        with np.load(one_hot_path) as oh_npz, np.load(contrib_path) as contrib_npz:
            one_hot_seqs = oh_npz["arr_0"]
            contribution_scores = contrib_npz["arr_0"]

        if one_hot_seqs.shape[2] < window:
            raise ValueError(
                f"Window ({window}) cannot be longer than the sequences ({one_hot_seqs.shape[2]})"
            )

        center = one_hot_seqs.shape[2] // 2
        start, end = _calculate_window_offsets(center, window)

        sequences = one_hot_seqs[:, :, start:end]
        attributions = contribution_scores[:, :, start:end]

        if top_n_regions is not None:
            top_n = min(max(top_n_regions, 1), len(sequences))
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
            logger.info(f"Modisco results already exist for class: {class_name}")

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


@log_and_raise(Exception)
def tfmodisco(
    contrib_dir: str | os.PathLike = "modisco_results",
    class_names: list[str] | None = None,
    output_dir: str | os.PathLike = "modisco_results",
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
    n_jobs: int = 1,
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
    n_jobs
        Number of classes to process in parallel. Each class is fully independent
        (own input/output files), so this scales near-linearly. ``1`` (default)
        runs serially. ``-1`` uses all available cores. Values other than ``1``
        require `joblib` and dispatch one worker process per class, with each
        worker pinned to a single BLAS/OpenMP/numba thread to avoid
        oversubscription.

    See Also
    --------
    crested.tl.contribution_scores_specific

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

    # Each class is fully independent (own input/output files), so we dispatch the
    # per-class work either serially (n_jobs == 1, the original behaviour) or
    # across worker processes (n_jobs != 1).
    run_kwargs = {
        "contrib_dir": contrib_dir,
        "output_dir": output_dir,
        "window": window,
        "max_seqlets": max_seqlets,
        "min_metacluster_size": min_metacluster_size,
        "min_final_cluster_size": min_final_cluster_size,
        "n_leiden": n_leiden,
        "report": report,
        "meme_db": meme_db,
        "verbose": verbose,
        "fdr": fdr,
        "sliding_window_size": sliding_window_size,
        "flank_size": flank_size,
        "top_n_regions": top_n_regions,
    }

    if n_jobs == 1:
        for class_name in class_names:
            _run_modisco_single_class(class_name, **run_kwargs)
    else:
        from joblib import Parallel, delayed, parallel_config

        logger.info(
            f"Running modisco for {len(class_names)} classes with n_jobs={n_jobs}"
        )
        # inner_max_num_threads pins each worker's BLAS/OpenMP threadpools to 1;
        # pin_numba_threads does the same for modiscolite's numba kernels.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            Parallel(n_jobs=n_jobs)(
                delayed(_run_modisco_single_class)(
                    class_name, pin_numba_threads=True, **run_kwargs
                )
                for class_name in class_names
            )


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
            "Pattern indices are specified, but no metacluster name is provided. Pattern indices will be ignored."
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
    sim_threshold: float = 7.0,
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

    all_patterns_list = [pat['pattern'] for pat in all_patterns.values()]
    sim_matrix1 = match_score_patterns(p, all_patterns_list)
    sim_matrix2 = match_score_patterns(all_patterns_list, p).T # for some reason changing the order can give different results
    sim_matrix = np.maximum(sim_matrix1, sim_matrix2)

    max_sim = np.max(sim_matrix)
    if max_sim > sim_threshold:
        match = True
        match_idx = np.argmax(sim_matrix[0])

    if not match:
        pattern_idx = len(all_patterns.keys())
        return add_pattern_to_dict(p, pattern_idx, cell_type, pos_pattern, all_patterns)

    if verbose:
        print(
            f"Match between {pattern_id} and {all_patterns[str(match_idx)]['pattern']['id']} with similarity score {max_sim:.2f}"
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
    return_info: bool = False,
) -> dict | tuple[dict, list[tuple[str, str, float]]]:
    """
    Double-checks the similarity of all patterns and merges them if they exceed the threshold.

    Filters out patterns with IC below the discard threshold at the last step and updates the keys.

    Parameters
    ----------
    all_patterns
        Dictionary of all patterns with metadata. Each pattern must include 'pattern', 'ic', and 'classes'.
    sim_threshold
        Similarity threshold for merging patterns.
    ic_discard_threshold
        IC threshold below which patterns are discarded unless they belong to multiple classes.
    verbose
        Flag to enable verbose output of merged patterns.
    return_info
        If True, also return a list of all performed merges as (pattern_id_1, pattern_id_2, similarity).

    Returns
    -------
    Updated patterns after merging and filtering with sequential keys.
    If `return_info=True`, also returns a list of performed merges.
    """
    current_meta = list(all_patterns.values())
    all_merges = []
    iteration = 0

    while True:
        iteration += 1
        N = len(current_meta)

        raw_patterns = [m["pattern"] for m in current_meta]
        raw_ids = [m["pattern"]["id"] for m in current_meta]

        S = match_score_patterns(raw_patterns, raw_patterns)
        S = np.maximum(S, S.T)
        np.fill_diagonal(S, -np.inf)

        candidates = np.argwhere(S > sim_threshold)
        candidates = [(i, j, S[i, j]) for i, j in candidates if i < j]

        if not candidates:
            if verbose:
                print(f"Iteration {iteration}: no more matches above {sim_threshold}")
            break

        candidates.sort(key=lambda x: x[2], reverse=True)

        matched = set()
        merges = []
        for i, j, score in candidates:
            if i in matched or j in matched:
                continue
            matched.add(i)
            matched.add(j)
            merges.append((i, j, score))
            all_merges.append((raw_ids[i], raw_ids[j], score))

        if verbose:
            print(f"Iteration {iteration}: performing {len(merges)} merges")
            for i, j, score in merges:
                print(f"  -> merging {raw_ids[i]} + {raw_ids[j]} (sim={score:.3f})")

        new_meta = []
        used = set()
        for i, j, _ in merges:
            merged = merge_patterns(current_meta[i], current_meta[j])
            new_meta.append(merged)
            used.update({i, j})

        for idx in range(N):
            if idx not in used:
                new_meta.append(current_meta[idx])

        current_meta = new_meta

    final = {}
    idx = 0
    for m in current_meta:
        if m["ic"] >= ic_discard_threshold or len(m["classes"]) > 1:
            final[str(idx)] = m
            idx += 1
        elif verbose:
            print(f"Dropping {m['pattern']['id']} (IC={m['ic']:.3f})")

    if verbose:
        print(f"Done after {iteration} iterations; {len(final)} patterns remain.")

    return (final, all_merges) if return_info else final


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


def _profile_cosine(contrib: np.ndarray, gex: np.ndarray, eps: float = 1e-9) -> float:
    """
    Uncentered cosine between a (non-negative) motif-usage profile and a TF-expression profile over cell types.

    Unlike Pearson correlation it does not mean-center, so a broad/flat expression profile scores
    low against a peaked contribution profile: breadth is penalized and alignment with the firing
    cell types is rewarded.
    """
    a = np.abs(np.asarray(contrib, dtype=float))
    b = np.clip(np.asarray(gex, dtype=float), 0, None)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > eps else 0.0


def _nnls_paralog_select(
    gex_raw: np.ndarray,
    contrib: np.ndarray,
    annots: list[str],
    *,
    zscore_threshold: float,
    correlation_threshold: float,
    alpha: float,
    keep_frac: float,
    rel_keep_frac: float,
) -> tuple[list[int], int]:
    """Deconvolution-based TF selection (``selection="nnls"`` in :func:`create_tf_ct_matrix`).

    A contribution profile is modelled as a non-negative combination of its candidate TFs'
    expression profiles, instead of scoring each TF in isolation. Three stages:

    1. **Pattern gate (Pearson):** keep a pattern only if at least one candidate has peaked
       expression (max z-score > ``zscore_threshold``) and correlates with the contribution
       (Pearson >= ``correlation_threshold``). Controls which patterns are annotated at all; it
       never thins the per-pattern candidate pool (that would starve the regression).
    2. **Full-pool non-negative ridge regression** per kept pattern: ``contrib ~ sum_t w_t * gex_t``,
       ``w_t >= 0`` (ridge ``alpha`` spreads weight over collinear columns), run on the FULL
       candidate pool so broad binders are competed against the specific ones and down-weighted.
       Keep ``w_t > 0`` and ``w_t >= keep_frac * max_t w``.
    3. **Expression-relevance gate (paralog selection):** the regression runs on peak-normalised
       (shape-only) expression, so a survivor may be barely expressed where the pattern actually
       fires. For each survivor compute a contribution-weighted mean RAW expression
       ``rel_t = (y . gex_raw_t) / sum(y)`` (``y = |contrib|``, so cell types where the pattern fires
       strongly count most) and keep it only if ``rel_t >= rel_keep_frac * max_t rel_t``. This drops
       barely-expressed candidates while keeping every paralog genuinely co-expressed in the firing
       cell types (no family grouping / name heuristic — co-expressed paralogs all pass, a weakly
       expressed one drops on its own).

    Expression is peak-normalised internally for the regression/gate (shape, not level); ``gex_raw``
    (raw level) drives the relevance gate. Returns ``(kept column indices into annots, number of
    columns kept by the regression before the relevance gate)``.
    """
    from scipy.optimize import nnls

    gex_norm = normalize_rows(gex_raw.T).T  # peak-normalise per TF for the regression + gate
    pid_of = [a.rsplit("_pattern_", 1)[1] for a in annots]

    # (1) pattern gate
    keep_pids = set()
    for col in range(len(annots)):
        g = gex_norm[:, col]
        if g.std() == 0:
            continue
        z = (g - g.mean()) / g.std()
        r = np.corrcoef(g, np.abs(contrib[:, col]))[0, 1]
        if np.max(z) > zscore_threshold and r >= correlation_threshold:
            keep_pids.add(pid_of[col])

    by_pid: dict[str, list[int]] = {}
    for col in range(len(annots)):
        by_pid.setdefault(pid_of[col], []).append(col)

    keep_idx: list[int] = []
    n_pre_relevance = 0  # columns kept by the regression, before the relevance gate
    for pid, cols in by_pid.items():
        if pid not in keep_pids:
            continue
        y = np.abs(contrib[:, cols[0]])  # contribution profile (shared across a pattern's columns)
        if y.sum() == 0:
            continue
        X = gex_norm[:, cols]
        if alpha > 0:  # non-negative Tikhonov ridge -> spreads weight over collinear paralogs
            n = X.shape[1]
            w = nnls(
                np.vstack([X, np.sqrt(alpha) * np.eye(n)]), np.concatenate([y, np.zeros(n)])
            )[0]
        else:
            w = nnls(X, y)[0]
        wmax = w.max()
        if wmax <= 0:
            continue
        kept = [k for k in range(len(cols)) if w[k] > 0 and w[k] >= keep_frac * wmax]
        n_pre_relevance += len(kept)
        # Expression-relevance gate: contribution-weighted mean RAW expression of each survivor,
        # keep those within rel_keep_frac of the pattern's best (drops barely-expressed candidates,
        # keeps co-expressed paralogs; no family grouping).
        rel = np.array([(y @ gex_raw[:, cols[k]]) / y.sum() for k in kept])
        relmax = rel.max() if len(rel) else 0.0
        if relmax > 0:
            kept = [k for k, rv in zip(kept, rel, strict=True) if rv >= rel_keep_frac * relmax]
        keep_idx.extend(cols[k] for k in kept)
    return sorted(keep_idx), n_pre_relevance


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
    cell_type: str, file_list: str | list[str], trim_ic_threshold: float, verbose: bool
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
            cell_type, matched_files[cell_type], trim_ic_threshold, verbose
        )
        all_trimmed_patterns += trimmed_patterns
        all_pattern_ids += pattern_ids

    # Add PPMs to each pattern
    pattern_ppms = [_pattern_to_ppm(p) for p in all_trimmed_patterns]
    for i, pat in enumerate(all_trimmed_patterns):
        pat["ppm"] = pattern_ppms[i]

    if verbose:
        print("Total patterns:", len(all_trimmed_patterns))

    # Compute pairwise TOMTOM similarity
    similarity_matrix = match_score_patterns(
        all_trimmed_patterns,
        all_trimmed_patterns,
        use_ppm=use_ppm,
        background_freqs=background_freqs,
    )

    # Construct output metadata dictionary
    pattern_dict = {
        pid: {
            "contrib_scores": all_trimmed_patterns[i]["contrib_scores"],
            "n_seqlets": all_trimmed_patterns[i]["seqlets"]["n_seqlets"],
        }
        for i, pid in enumerate(all_pattern_ids)
    }

    return similarity_matrix, all_pattern_ids, pattern_dict


def process_patterns(
    matched_files: dict[str, str | list[str] | None],
    sim_threshold: float = 6.0,
    trim_ic_threshold: float = 0.025,
    discard_ic_threshold: float = 0.1,
    clustering: str | None = None,
    linkage_method: str = "average",
    sort_by: str | None = "n_seqlets",
    representative: str = "n_seqlets",
    verbose: bool = False,
) -> dict[str, dict[str, str | list[float]]]:
    """
    Process genomic patterns from matched HDF5 files, trim based on information content, and match to known patterns.

    Parameters
    ----------
    matched_files
        dictionary with class names as keys and paths to HDF5 files as values.
    sim_threshold
        Similarity threshold (``-log10(pval)`` from TOMTOM, memesuite-lite) for grouping patterns into a cluster.
    trim_ic_threshold
        Information content threshold for trimming patterns.
    discard_ic_threshold
        Information content threshold for discarding patterns.
    clustering
        How to group patterns across cell types into clusters.
        ``"agglomerative"`` (the default) computes the full pairwise similarity
        once and runs deterministic, order-independent hierarchical clustering (see
        ``linkage_method``) with a single cut at ``sim_threshold`` — same output
        structure, reproducible regardless of input order.
        ``"greedy"`` is the original order-dependent leader clustering (assign each
        pattern to the best existing cluster above ``sim_threshold``, else start a new
        one) followed by a post-hoc all-vs-all merge; use it to reproduce analyses run
        before the default changed.
        If left unset (``None``), defaults to ``"agglomerative"`` and emits a warning
        noting the changed default.
    linkage_method
        Linkage for ``clustering="agglomerative"`` (any `scipy.cluster.hierarchy.linkage`
        method, e.g. ``"average"``, ``"complete"``, ``"single"``). Ignored for greedy.
        ``"average"`` (default) controls chaining better than single linkage.
    sort_by
        How to order the returned clusters (and their string keys, ``"0"``, ``"1"``, ...).
        ``"n_seqlets"`` (default) sorts by descending total seqlet count summed over a
        cluster's classes, so ``"0"`` is the most-supported cluster. ``"ic"`` sorts by
        descending cluster information content. ``None`` keeps the internal
        insertion/merge order. Applied to both clustering methods.
    representative
        Which member instance to use as a cluster's representative motif (the logo shown
        and the PPM matched against the motif database for TF assignment). ``"n_seqlets"``
        (default) picks the most-supported instance (most seqlets) — robust, since it can't
        be dragged to a single noisy long outlier. ``"ic_total"`` picks the most *complete*
        motif by summed per-position IC (= mean IC x length); fuller motif, more TOMTOM
        columns, but noisier on ragged/over-merged clusters. ``"ic_mean"`` is the legacy
        mean per-position IC (favours short, tight motifs); use it only to reproduce
        pre-change runs. Agglomerative clustering only (greedy derives its representative
        during matching).
    verbose
        Flag to enable verbose output.

    See Also
    --------
    crested.tl.modisco.match_h5_files_to_classes

    Returns
    -------
    All processed patterns with metadata.
    """
    if clustering is None:
        clustering = "agglomerative"
        logger.warning(
            "`process_patterns` now defaults to clustering='agglomerative' "
            "(deterministic, order-independent) instead of 'greedy'. "
            "Results may differ from earlier runs; pass clustering='greedy' to reproduce them."
        )

    if representative not in ("ic_total", "n_seqlets", "ic_mean"):
        raise ValueError(
            f"representative must be 'ic_total', 'n_seqlets' or 'ic_mean', got '{representative}'"
        )

    if clustering == "agglomerative":
        all_patterns = _process_patterns_agglomerative(
            matched_files,
            sim_threshold=sim_threshold,
            trim_ic_threshold=trim_ic_threshold,
            discard_ic_threshold=discard_ic_threshold,
            linkage_method=linkage_method,
            representative=representative,
            verbose=verbose,
        )
    elif clustering == "greedy":
        all_patterns = {}

        for cell_type in matched_files:
            trimmed_patterns, pattern_ids, is_pattern_pos = _read_and_trim_patterns(
                cell_type, matched_files[cell_type], trim_ic_threshold, verbose
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
    else:
        raise ValueError(
            f"clustering must be 'greedy' or 'agglomerative', got '{clustering}'"
        )

    return _sort_patterns(all_patterns, sort_by)


def _sort_patterns(all_patterns: dict, sort_by: str | None) -> dict:
    """Reorder clusters and re-key them ``"0"``..``"N-1"`` according to `sort_by`.

    `sort_by` is one of ``"n_seqlets"`` (descending total seqlets over a cluster's
    classes), ``"ic"`` (descending cluster IC), or ``None`` (keep current order).
    """
    if sort_by is None:
        return all_patterns

    if sort_by == "n_seqlets":
        def key(pat):
            return sum(c["n_seqlets"] for c in pat["classes"].values())
    elif sort_by == "ic":
        def key(pat):
            return pat["ic"]
    else:
        raise ValueError(
            f"sort_by must be 'n_seqlets', 'ic', or None, got '{sort_by}'"
        )

    ordered = sorted(all_patterns.values(), key=key, reverse=True)
    return {str(i): pat for i, pat in enumerate(ordered)}


def _representative_key(m: dict, representative: str):
    """Sort key selecting a cluster's representative instance (higher = preferred).

    ``"ic_total"`` = most *complete* motif (summed per-position IC = mean IC x length, so
    extra columns only help when they carry information); ``"n_seqlets"`` = most-supported
    instance (most seqlets); ``"ic_mean"`` = legacy mean per-position IC, which favours
    short, tight motifs.
    """
    if representative == "n_seqlets":
        return m["n_seqlets"]
    if representative == "ic_mean":
        return m["ic"]
    return m["ic"] * len(m["ppm"])  # "ic_total": mean per-position IC x n_positions


def _process_patterns_agglomerative(
    matched_files: dict[str, str | list[str] | None],
    sim_threshold: float,
    trim_ic_threshold: float,
    discard_ic_threshold: float,
    linkage_method: str = "average",
    representative: str = "n_seqlets",
    verbose: bool = False,
) -> dict[str, dict]:
    """Deterministic, order-independent alternative to the greedy clustering in `process_patterns`.

    Reads and trims all patterns, computes the full symmetrized pairwise TOMTOM
    similarity once, then clusters with hierarchical clustering cut at
    ``sim_threshold``. Produces the same `all_patterns` structure as the greedy
    path (representative selected by `representative`; per-class seqlet counts
    summed) and applies the same low-IC single-class discard filter.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    # 1. Read, trim and prepare every pattern (mirrors the per-pattern prep in match_to_patterns).
    patterns: list[dict] = []
    for cell_type in matched_files:
        trimmed, ids, is_pos = _read_and_trim_patterns(
            cell_type, matched_files[cell_type], trim_ic_threshold, verbose
        )
        for p, pid, pos in zip(trimmed, ids, is_pos, strict=True):
            p["id"] = pid
            p["pos_pattern"] = pos
            p["n_seqlets"] = p["seqlets"]["n_seqlets"][0]
            ppm = _pattern_to_ppm(p)
            _, ic_pos, _ = compute_ic(ppm)
            p["ppm"] = ppm
            p["ic"] = np.mean(ic_pos)
            p["class"] = cell_type
            patterns.append(p)

    if len(patterns) == 0:
        return {}

    # 2. Full pairwise similarity (-log10 p), symmetrized, then hierarchical clustering.
    if len(patterns) == 1:
        labels = np.array([1])
    else:
        sim = match_score_patterns(patterns, patterns)
        sim = np.maximum(sim, sim.T)
        max_sim = float(np.max(sim))
        dist = max_sim - sim  # higher similarity -> smaller distance
        dist = np.clip((dist + dist.T) / 2, 0, None)
        np.fill_diagonal(dist, 0.0)
        z = linkage(squareform(dist, checks=False), method=linkage_method)
        # Cut so that clusters are formed below a linkage distance corresponding to
        # sim_threshold (e.g. average linkage -> average pairwise similarity > sim_threshold).
        labels = fcluster(z, t=max_sim - sim_threshold, criterion="distance")

    # 3. Build the all_patterns dict per cluster (order-independent within a cluster:
    #    representative selected by `representative`, per-class seqlet counts are summed).
    all_patterns: dict[str, dict] = {}
    out_idx = 0
    for lab in np.unique(labels):
        members = [patterns[i] for i in np.nonzero(labels == lab)[0]]
        rep = max(members, key=lambda m: _representative_key(m, representative))
        entry = {
            "pattern": rep,
            "pos_pattern": rep["pos_pattern"],
            "ppm": rep["ppm"],
            "ic": rep["ic"],
            "instances": {m["id"]: m for m in members},
            "classes": {},
        }
        # Preserve first-seen class order, summing seqlets and keeping the representative instance.
        seen_classes: list[str] = []
        for m in members:
            if m["class"] not in seen_classes:
                seen_classes.append(m["class"])
        for ct in seen_classes:
            ct_members = [m for m in members if m["class"] == ct]
            ct_rep = max(ct_members, key=lambda m: _representative_key(m, representative))
            ct_rep["n_seqlets"] = int(sum(m["n_seqlets"] for m in ct_members))
            entry["classes"][ct] = ct_rep

        # Same discard rule as post_hoc_merging: drop low-IC patterns unless multi-class.
        if entry["ic"] >= discard_ic_threshold or len(entry["classes"]) > 1:
            all_patterns[str(out_idx)] = entry
            out_idx += 1
        elif verbose:
            print(f"Dropping {rep['id']} (IC={entry['ic']:.3f})")

    if verbose:
        print(
            f"Agglomerative ({linkage_method}) clustering: {len(patterns)} patterns "
            f"-> {len(all_patterns)} clusters (cut at sim_threshold={sim_threshold})"
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
    crested.pl.modisco.clustermap

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
    crested.pl.modisco.similarity_heatmap
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
        Modisco output directory, containing folders with per-class reports that have a trimmed_logos directory.

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
        im_path = os.path.join(
            contribution_dir,
            f"{pattern_class}_report/trimmed_logos/{pos_neg}pattern_{id_split[-1]}.cwm.fwd.png",
        )
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
        Modisco output directory, containing folders with per-class reports.

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
    all_patterns: dict, html_paths: list[str], p_val_thr: float = 0.05, q_val_thr: str = 'deprecated'
) -> dict[int, dict[str, list[str]]]:
    """
    Find and filter pattern matches from the modisco-lite list of patterns to the motif database from the corresponding HTML paths.

    Parameters
    ----------
    all_patterns
        A dictionary of patterns with metadata.
    html_paths
        A list of file paths to HTML files containing motif databases.
    p_val_thr
        The threshold for p-value filtering if a q-value or p-value column is present. Default is 0.05.

    Returns
    -------
    A dictionary with pattern indices as keys and a dictionary of matches as values.
    """
    if q_val_thr != 'deprecated':
        p_val_thr = q_val_thr
        logger.warning(f"Modisco renamed the `qval` column to `pval`, so `q_val_thr` is now called `p_val_thr` as well. Please use `p_val_thr={q_val_thr}`.")
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
                        pval_column = f"pval{j}"
                        qval_column = f"qval{j}"
                        match_column = f"match{j}"
                        pval = None
                        if pval_column in matching_row.columns and match_column in matching_row.columns:
                            pval = matching_row[pval_column].values[0]
                        elif qval_column in matching_row.columns and match_column in matching_row.columns:
                            pval = matching_row[qval_column].values[0]
                        if pval is not None and pval < p_val_thr:
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
    similarity_metric: str = "pearson",
    selection: str = "nnls",
    nnls_alpha: float = 1.0,
    nnls_keep_frac: float = 0.2,
    rel_keep_frac: float = 0.1,
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
        Minimum expression a TF must reach in at least one cell type where the pattern fires (contribution > `importance_threshold`) to be kept. Applied to the expression as provided in `df` (mean per cell type from your adata, whatever its matrix holds — not necessarily raw counts), *before* this function's `normalize_gex` peak-scaling (which scales each TF to max 1 and would make a level floor meaningless). Default 0 (no floor). Note that some genuine TFs are lowly expressed, so raising this can drop real candidates; with `verbose=True` the per-candidate expression percentiles on the firing cell types are printed to help calibrate.
    importance_threshold
        The minimum pattern importance value. Default is 0.
    pattern_parameter
        Parameter which is used to indicate the pattern's importance. Either average contribution score ('contrib'), or number of pattern instances ('seqlet_count', default) and its log ('seqlet_count_log').
    filter_correlation
        Whether to filter based on Pearson correlation between `tf_gex` and `ct_contribs`. Default is False.
    zscore_threshold
        Zscore used for filtering TF candidates. If the max zscore over the cell types is belofw this threshold, the TF gets discarded. Default is 2.
    correlation_threshold
        Minimum agreement (see `similarity_metric`) between expression and contribution profile required to keep a column if filtering is enabled. Default is 0.2.
    similarity_metric
        Metric for the expression/contribution agreement, used by the ``"threshold"`` selection's correlation filter and by the ``"nnls"`` selection's pattern gate. ``"pearson"`` (default) keeps the original mean-centered Pearson correlation. ``"cosine"`` uses an uncentered cosine between the absolute contribution and the (non-negative) expression profile. Because it is not mean-centered, cosine matches the *shape* of the two profiles: a broadly-expressed TF on a broad contribution still scores high, but a broadly-expressed TF on a cell-type-specific contribution scores low and is dropped. Recommended (with a slightly higher `correlation_threshold`, ~0.55, since cosine has a positive floor) when broadly-expressed TFs are being selected for specific patterns. (The ``"nnls"`` pattern gate always uses Pearson regardless of this value.)
    selection
        How candidate TF-pattern columns are selected. ``"nnls"`` (default) = deconvolution: a Pearson pattern gate decides which patterns are annotated, then a non-negative ridge regression models each pattern's contribution as a combination of its full candidate pool (down-weighting broadly-expressed binders that merely correlate), and finally an expression-relevance gate (`rel_keep_frac`) keeps the candidates actually expressed where the pattern fires. See `crested.tl.modisco._tfmodisco._nnls_paralog_select`. With ``"nnls"`` the `filter_correlation` flag is ignored (the pattern gate uses `zscore_threshold` + `correlation_threshold`). ``"threshold"`` = the original per-column gates (gex/importance, then the `filter_correlation` correlation/zscore gate); use it to reproduce pre-change analyses.
    nnls_alpha
        ``selection="nnls"`` only. Non-negative ridge (Tikhonov) strength for the per-pattern regression. >0 spreads weight across collinear candidate columns so a single paralog does not arbitrarily absorb it; ~1.0 is a good default. Default 1.0.
    nnls_keep_frac
        ``selection="nnls"`` only. Within a pattern, keep a TF if its regression weight is > 0 and >= ``nnls_keep_frac`` x the pattern's maximum weight. Default 0.2.
    rel_keep_frac
        ``selection="nnls"`` only. Expression-relevance gate applied to the regression survivors: for each survivor compute a contribution-weighted mean RAW expression ``rel_t = (|contrib| . gex_t) / sum(|contrib|)`` (cell types where the pattern fires strongly dominate), and keep it only if ``rel_t >= rel_keep_frac`` x the pattern's best survivor. This drops candidates barely expressed where the pattern fires while keeping every genuinely co-expressed paralog (no family grouping or name heuristic — 1 or more paralogs survive on their own merit). Higher = stricter (fewer paralogs). Default 0.1.
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

    df = df.reindex(classes)  # Ensure they are in same order.

    if pattern_parameter not in ["contrib", "seqlet_count", "seqlet_count_log"]:
        logger.info("Pattern parameter not valid. Setting to default ('seqlet_count').")
        pattern_parameter = "seqlet_count"

    if selection not in ("threshold", "nnls"):
        logger.info("selection not valid. Setting to default ('nnls').")
        selection = "nnls"

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
    # Keep the input expression (as given in df) for the level-based min_tf_gex gate: normalize_gex
    # below peak-scales each TF to max 1, which would make an absolute expression floor meaningless.
    input_gex = tf_ct_matrix[:, :, 0].copy()
    if normalize_pattern_importances:
        tf_ct_matrix[:, :, 1] = normalize_rows(tf_ct_matrix[:, :, 1])
    if normalize_gex:
        tf_ct_matrix[:, :, 0] = normalize_rows(tf_ct_matrix[:, :, 0].T).T

    # Per-step overview of how many TF-pattern candidate columns survive each filter.
    n_skipped_no_gex = total_tf_patterns - len(tf_pattern_annots)
    if verbose:
        print("create_tf_ct_matrix - TF-pattern candidate filtering:")
        print(
            f"  candidates (TF-motif pairs) : {len(tf_pattern_annots):4d}"
            f"  (skipped {n_skipped_no_gex}: TF absent from expression table)"
        )

    # Keep columns where the TF reaches min_tf_gex (input expression, before normalize_gex) in at
    # least one cell type where the pattern fires (contribution > importance_threshold).
    initial_columns = tf_ct_matrix.shape[1]
    columns_to_keep = []
    peak_input_gex = []  # max input expression on the firing CTs per candidate, for verbose calibration

    for col in range(initial_columns):
        input_gex_col = input_gex[:, col]
        ct_contribs_col = tf_ct_matrix[:, col, 1]

        # Identify relevant ct_contribs
        relevant_contribs = ct_contribs_col > importance_threshold

        if np.any(relevant_contribs):
            peak_input_gex.append(float(np.max(input_gex_col[relevant_contribs])))

        # Check if there are valid ct_contribs and (input) tf_gex above the threshold
        if np.any(relevant_contribs) and np.any(
            input_gex_col[relevant_contribs] > min_tf_gex
        ):
            columns_to_keep.append(col)

    # Convert columns_to_keep to a boolean mask
    columns_to_keep = np.array(columns_to_keep)

    # Filter the matrix and annotations based on the columns_to_keep
    final_columns = len(columns_to_keep)
    removed_columns = initial_columns - final_columns

    tf_ct_matrix = tf_ct_matrix[:, columns_to_keep, :]
    input_gex = input_gex[:, columns_to_keep]  # keep aligned (used by selection="nnls")
    tf_pattern_annots = [
        annot for i, annot in enumerate(tf_pattern_annots) if i in columns_to_keep
    ]

    if verbose:
        print(
            f"  after gex/importance gate   : {final_columns:4d}  (dropped {removed_columns};"
            f" min_tf_gex={min_tf_gex} on input expr, importance_threshold={importance_threshold})"
        )
        if peak_input_gex:
            p10, p50, p90 = np.percentile(peak_input_gex, [10, 50, 90])
            print(
                f"      input expr on firing CTs (peak/candidate): "
                f"p10={p10:.3g} p50={p50:.3g} p90={p90:.3g}  -> calibrate min_tf_gex here"
            )

    # TF-pattern column selection.
    if selection == "nnls":
        before = tf_ct_matrix.shape[1]
        keep_idx, n_pre_relevance = _nnls_paralog_select(
            input_gex,
            tf_ct_matrix[:, :, 1],
            tf_pattern_annots,
            zscore_threshold=zscore_threshold,
            correlation_threshold=correlation_threshold,
            alpha=nnls_alpha,
            keep_frac=nnls_keep_frac,
            rel_keep_frac=rel_keep_frac,
        )
        tf_ct_matrix = tf_ct_matrix[:, keep_idx, :]
        tf_pattern_annots = [tf_pattern_annots[i] for i in keep_idx]
        if verbose:
            print(
                f"  after nnls regression       : {n_pre_relevance:4d}  (dropped {before - n_pre_relevance};"
                f" alpha={nnls_alpha}, keep_frac={nnls_keep_frac})"
            )
            print(
                f"  after relevance gate        : {len(keep_idx):4d}"
                f"  (dropped {n_pre_relevance - len(keep_idx)}; rel_keep_frac={rel_keep_frac})"
            )

    # Filter out TF candidates for patterns that do not show correlation between their expression and importance profiles.
    elif filter_correlation:
        if similarity_metric not in ("pearson", "cosine"):
            logger.info(
                "similarity_metric not valid. Setting to default ('pearson')."
            )
            similarity_metric = "pearson"

        initial_columns = tf_ct_matrix.shape[1]
        columns_to_keep = []

        for col in range(initial_columns):
            tf_gex_col = tf_ct_matrix[:, col, 0]
            ct_contribs_col = np.abs(tf_ct_matrix[:, col, 1])

            tf_gex_col_z = (tf_gex_col - np.mean(tf_gex_col)) / np.std(tf_gex_col)

            if similarity_metric == "cosine":
                score = _profile_cosine(ct_contribs_col, tf_gex_col)
            else:
                score = np.corrcoef(tf_gex_col, ct_contribs_col)[0, 1]

            if (np.max(tf_gex_col_z) > zscore_threshold) and (
                score >= correlation_threshold
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
            print(
                f"  after {similarity_metric} gate ({correlation_threshold}) : {final_columns:4d}"
                f"  (dropped {removed_columns}; zscore_threshold={zscore_threshold})"
            )

    if verbose:
        print(f"  -> kept {len(tf_pattern_annots)} TF-pattern columns")

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
    from scipy import sparse

    # Read the AnnData object from the specified H5AD file
    adata = anndata.read_h5ad(file_path)

    # CPM normalize the counts if necessary (in-place, sparse-friendly)
    if cpm_normalize:
        sc.pp.normalize_total(adata)

    # Check if the specified cell type column exists in the cell metadata
    if cell_type_column not in adata.obs.columns:
        raise ValueError(f"Column '{cell_type_column}' not found in cell metadata")

    # Group cells by cell type, dropping unlabelled cells (matches the previous
    # pandas-groupby behaviour, which excluded NaN groups).
    labels = adata.obs[cell_type_column].astype("category")
    codes = labels.cat.codes.to_numpy()
    categories = labels.cat.categories
    valid = codes >= 0  # -1 == NaN / unlabelled

    # Compute the per-cell-type mean directly on (possibly sparse) X via an
    # indicator-matrix product, so the full matrix is never densified:
    #     sums  = indicator @ X        # (n_groups, n_genes)
    #     means = sums / counts
    # Keeps memory at O(n_groups * n_genes) instead of O(n_cells * n_genes),
    # which matters for atlas-scale inputs (e.g. ~750k cells would be ~38 GB dense).
    indicator = sparse.csr_matrix(
        (
            np.ones(int(valid.sum()), dtype="float64"),
            (codes[valid], np.nonzero(valid)[0]),
        ),
        shape=(len(categories), adata.n_obs),
    )
    counts = np.asarray(indicator.sum(axis=1)).ravel()

    sums = indicator @ adata.X  # sparse @ (sparse | dense)
    sums = np.asarray(sums.todense()) if sparse.issparse(sums) else np.asarray(sums)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts[:, None]  # groups with 0 cells -> NaN, dropped below

    mean_expression_per_cell_type = pd.DataFrame(
        means,
        index=pd.Index(categories, name=cell_type_column),
        columns=adata.var_names,
    )
    # Drop categories with no cells (unused categorical levels) and sort by cell
    # type, matching the previous groupby output.
    mean_expression_per_cell_type = mean_expression_per_cell_type.loc[
        counts > 0
    ].sort_index()

    return mean_expression_per_cell_type
