"""Utility functions specific for the TL module."""

from __future__ import annotations

import numpy as np


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
