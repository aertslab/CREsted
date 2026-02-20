from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class EnhancerOptimizer:
    """
    Class to optimize the mutated sequence based on the original prediction.

    Can be passed as the 'enhancer_optimizer' argument to :func:`crested.tl.design.in_silico_evolution`

    Parameters
    ----------
    optimize_func
        Function to optimize the mutated sequence based on the original prediction.

    See Also
    --------
    crested.tl.design.in_silico_evolution
    """

    def __init__(self, optimize_func: Callable[..., int]) -> None:
        """Initialize the EnhancerOptimizer class."""
        self.optimize_func = optimize_func

    def get_best(
        self,
        mutated_predictions: np.ndarray,
        original_prediction: np.ndarray,
        target: int | np.ndarray,
        **kwargs: dict[str, Any],
    ) -> int:
        """
        Get the index of the best mutated sequence based on the original prediction.

        Parameters
        ----------
        mutated_predictions
            The predictions of the mutated sequences.
        original_prediction
            The prediction of the original sequence.
        target
            An integer or array representing some target to optimize for.
            For example, this can be the target class index, or an array of target values to maximize.
            Depends on the optimization function used.
        **kwargs
            Additional keyword arguments to pass to the optimization function.
        """
        return self.optimize_func(
            mutated_predictions, original_prediction, target, **kwargs
        )

def _weighted_difference(
    mutated_predictions: np.ndarray,
    original_prediction: np.ndarray,
    target: int,
    class_penalty_weights: np.ndarray | None = None,
):
    if len(original_prediction.shape) == 1:
        original_prediction = original_prediction[None]
    n_classes = original_prediction.shape[1]
    penalty_factor = 1 / n_classes

    target_increase = mutated_predictions[:, target] - original_prediction[:, target]
    other_increases = mutated_predictions - original_prediction

    other_increases[:, target] = 0

    if class_penalty_weights is None:
        class_penalty_weights = np.ones(n_classes)

    score = target_increase - penalty_factor * np.sum(
        other_increases * class_penalty_weights, axis=1
    )

    return np.argmax(score)

def derive_intermediate_sequences(
    enhancer_design_intermediate: dict,
) -> list[list[str]]:
    """
    Derive the intermediate sequences from the enhancer design intermediate dictionary.

    Parameters
    ----------
    enhancer_design_intermediate
        The enhancer design intermediate dictionary.

    Returns
    -------
    A nested list of intermediate sequences, of shape [[seq1_step1, seq1_step2, ...], [seq2_step1, seq2_step2, ...]]

    See Also
    --------
    crested.tl.design.in_silico_evolution
    crested.tl.design.motif_insertion
    """
    all_designed_list = []
    for intermediate_dict in enhancer_design_intermediate:
        current_sequence = intermediate_dict["initial_sequence"]
        sequence_list = [current_sequence]
        for loc, change in intermediate_dict["changes"]:
            if loc == -1:
                continue
            else:
                current_sequence = (
                    current_sequence[:loc]
                    + change
                    + current_sequence[loc + len(change) :]
                )
                sequence_list.append(current_sequence)
        all_designed_list.append(sequence_list)
    return all_designed_list


def create_random_sequences(
    n_sequences: int, seq_len: int, acgt_distribution: np.ndarray | None
) -> np.ndarray:
    """Create random sequences based on the given ACGT distribution."""
    random_sequences = np.empty((n_sequences), dtype=object)
    if acgt_distribution is None:
        acgt_distribution = np.full((seq_len, 4), 0.25)  # shape: (seq_len, 4)
    if acgt_distribution.shape == (4,):
        acgt_distribution = np.full((seq_len, 4), acgt_distribution)
    for idx_seq in range(n_sequences):
        current_sequence = []
        for idx_loc in range(seq_len):
            current_sequence.append(
                np.random.choice(
                    ["A", "C", "G", "T"], p=list(acgt_distribution[idx_loc])
                )
            )
        random_sequences[idx_seq] = "".join(current_sequence)

    return random_sequences  # shape (N, L)


def parse_starting_sequences(starting_sequences) -> np.ndarray:
    """Convert starting sequences to expected array format."""
    if isinstance(starting_sequences, str):
        starting_sequences = [starting_sequences]

    n_sequences = len(starting_sequences)
    starting_sequences_array = np.empty((n_sequences), dtype=object)
    for idx, sequence in enumerate(starting_sequences):
        starting_sequences_array[idx] = sequence

    return starting_sequences_array  # shape (N, L)


def generate_motif_insertions(x, motif, flanks=(0, 0), masked_locations=None):
    """Generate motif insertions in a sequence."""
    _, L, A = x.shape
    start, end = 0, L
    x_mut = []
    motif_length = motif.shape[1]
    start = flanks[0]
    end = L - flanks[1] - motif_length + 1
    insertion_locations = []

    for motif_start in range(start, end):
        motif_end = motif_start + motif_length
        if masked_locations is not None:
            if np.any(
                (motif_start <= masked_locations) & (masked_locations < motif_end)
            ):
                continue
        x_new = np.copy(x)
        x_new[0, motif_start:motif_end, :] = motif
        x_mut.append(x_new)
        insertion_locations.append(motif_start)

    return np.concatenate(x_mut, axis=0), insertion_locations
