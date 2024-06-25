"""Code adapted from https://github.com/jmschrei/tfmodisco-lite/blob/main/modisco."""

from __future__ import annotations

import os

import anndata
import modiscolite
import numpy as np
import re
from loguru import logger

from crested._logging import log_and_raise


def _calculate_window_offsets(center: int, window_size: int) -> tuple:
    return (center - window_size // 2, center + window_size // 2)


@log_and_raise(Exception)
def tfmodisco(
    contrib_dir: os.PathLike = 'modisco_results',
    class_names: list[str] | None = None,
    output_dir: os.PathLike = "modisco_results",
    max_seqlets: int = 2000,
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
        List of class names to process. If None, all class names found in the output directory will be processed.
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
    >>> evaluator.calculate_contribution_scores(
    ...     adata, class_names=["Astro", "Vip"], method="integrated_grad"
    ... )
    >>> crested.tl.tfmodisco(
    ...     adata, class_names=["Astro", "Vip"], output_dir="modisco_results"
    ... )
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use all class names found in the contribution directory if class_names is not provided
    if class_names is None:
        class_names = [re.match(r'(.+?)_oh\.npz$', f).group(1) for f in os.listdir(contrib_dir) if f.endswith('_oh.npz')]
        class_names = list(set(class_names))

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

            sequences = sequences.transpose(0,2,1)
            attributions = attributions.transpose(0,2,1)

            sequences = sequences.astype("float32")
            attributions = attributions.astype("float32")

            # Define filenames for the output files
            output_file = os.path.join(output_dir, f"{class_name}_modisco_results.h5")
            report_dir = os.path.join(output_dir, f"{class_name}_report")

            # Check if the modisco results .h5 file does not exist for the class
            if not os.path.exists(output_file):
                logger.info(f"Running modisco for class: {class_name}")
                pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
                    hypothetical_contribs=attributions,
                    one_hot=sequences,
                    max_seqlets_per_metacluster=max_seqlets,
                    sliding_window_size=20,
                    flank_size=5,
                    target_seqlet_fdr=0.05,
                    n_leiden_runs=n_leiden,
                    verbose=verbose,
                )

                modiscolite.io.save_hdf5(output_file, pos_patterns, neg_patterns, window_size=window)

                # Generate the modisco report
                if report:
                    modiscolite.report.report_motifs(
                        output_file,
                        report_dir,
                        meme_motif_db=meme_db,
                        top_n_matches=3,
                    )
            else:
                print(f"Modisco results already exist for class: {class_name}")

        except KeyError as e:
            logger.error(f"Missing data for class: {class_name}, error: {e}")
