"""Tooling kit that handles predictions, contribution scores, enhancer design, ... ."""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
from loguru import logger
from tqdm import tqdm

from crested.tl.design._utils import (
    EnhancerOptimizer,
    _weighted_difference,
    create_random_sequences,
    generate_motif_insertions,
    parse_starting_sequences,
)
from crested.utils._seq_utils import (
    generate_mutagenesis,
    hot_encoding_to_sequence,
    one_hot_encode_sequence,
)


def in_silico_evolution(
    n_mutations: int,
    target: int | np.ndarray,
    model: keras.Model | list[keras.Model],
    n_sequences: int = 1,
    return_intermediate: bool = False,
    no_mutation_flanks: tuple[int, int] | None = None,
    target_len: int | None = None,
    enhancer_optimizer: EnhancerOptimizer | None = None,
    starting_sequences: str | list | None = None,
    acgt_distribution: np.ndarray[float] | None = None,
    **kwargs: dict[str, Any],
) -> list | tuple[list[dict], list]:
    """
    Create synthetic enhancers for a specified class using in silico evolution (ISE).

    Parameters
    ----------
    n_mutations
        Number of mutations to make in each sequence.
        20 is a good starting point for most cases.
    target
        Using the default weighted_difference optimization function this should be the index of the target class to design enhancers for.
        This gets passed to the `get_best` function of the EnhancerOptimizer, so can represent other target values too.
    model
        A (list of) trained keras model(s) to design enhancers with.
        If a list of models is provided, the predictions will be averaged across all models.
    n_sequences
        Number of enhancers to design
    return_intermediate
        If True, returns a dictionary with predictions and changes made in intermediate steps for selected
        sequences, which can be visualized in :func:`~crested.pl.design.step_predictions` and :func:`~crested.pl.design.step_contribution_scores`.
    no_mutation_flanks
        A tuple of integers which determine the regions in each flank to not do insertions.
    target_len
        Length of the area in the center of the sequence to make mutations in.
        Ignored if no_mutation_flanks is provided.
    acgt_distribution
        An array of floats representing the distribution of A, C, G, and T in the genome (in that order).
        If the array is of shape (L, 4), it will be assumed to be per position. If it is of shape (4,), it will be assumed to be overall.
        If None, a uniform distribution will be used.
        This will be used to generate random sequences if starting_sequences is not provided.
        You can calculate these using :func:`~crested.utils.calculate_nucleotide_distribution`.
    kwargs
        Keyword arguments that will be passed to the `get_best` function of the EnhancerOptimizer

    Returns
    -------
    A list of designed sequences. If return_intermediate is True, will also return a list of dictionaries of intermediate
    mutations and predictions.

    See Also
    --------
    crested.utils.calculate_nucleotide_distribution
    crested.tl.design.EnhancerOptimizer
    crested.pl.design.step_predictions
    crested.pl.design.step_contribution_scores

    Examples
    --------
    >>> acgt_distribution = crested.utils.calculate_nucleotide_distribution(
    ...     adata, genome, per_position=True
    ... )  # shape (L, 4)
    >>> target_idx = adata.obs_names.index("my_celltype")
    >>> intermediate_results, designed_sequences = crested.tl.design.in_silico_evolution(
    ...     n_mutations=20,
    ...     target=target_idx,
    ...     model=my_trained_model,
    ...     n_sequences=1,
    ...     return_intermediate=True,
    ...     acgt_distribution=acgt_distribution,
    ... )
    """
    if enhancer_optimizer is None:
        enhancer_optimizer = EnhancerOptimizer(optimize_func=_weighted_difference)
    if not isinstance(model, list):
        model = [model]
    seq_len = model[0].input_shape[1]

    # determine the flanks without changes
    if no_mutation_flanks is not None and target_len is not None:
        logger.warning(
            "Both no_mutation_flanks and target_len set, using no_mutation_flanks."
        )
    elif no_mutation_flanks is None and target_len is not None:
        if (seq_len - target_len) % 2 == 0:
            no_mutation_flanks = (
                int((seq_len - target_len) // 2),
                int((seq_len - target_len) // 2),
            )
        else:
            no_mutation_flanks = (
                int((seq_len - target_len) // 2),
                int((seq_len - target_len) // 2) + 1,
            )

    elif no_mutation_flanks is None and target_len is None:
        no_mutation_flanks = (0, 0)

    # create initial sequences
    if starting_sequences is None:
        if acgt_distribution is None:
            logger.warning(
                "No nucleotide distribution provided. Using uniform distribution."
            )
        initial_sequences = create_random_sequences(
            n_sequences=n_sequences,
            seq_len=seq_len,
            acgt_distribution=acgt_distribution,
        )
    else:
        initial_sequences = parse_starting_sequences(starting_sequences)
        n_sequences = initial_sequences.shape[0]

    # initialize
    designed_sequences: list[str] = []
    intermediate_info_list: list[dict] = []

    sequence_onehot_prev_iter = np.zeros((n_sequences, seq_len, 4), dtype=np.uint8)

    # calculate total number of mutations per sequence
    _, L, A = sequence_onehot_prev_iter.shape
    start, end = 0, L
    start = no_mutation_flanks[0]
    end = L - no_mutation_flanks[1]
    TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ = (end - start) * (A - 1)

    mutagenesis = np.zeros((n_sequences, TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ, seq_len, 4))

    for i, sequence in enumerate(initial_sequences):
        sequence_onehot_prev_iter[i] = one_hot_encode_sequence(sequence)

    for _iter in tqdm(range(n_mutations)):
        baseline_prediction = []
        for m in model:
            baseline_prediction.append(
                m.predict(sequence_onehot_prev_iter, verbose=False)
            )
        baseline_prediction = np.mean(baseline_prediction, axis=0)
        if _iter == 0:
            for i in range(n_sequences):
                # initialize info
                intermediate_info_list.append(
                    {
                        "initial_sequence": hot_encoding_to_sequence(
                            sequence_onehot_prev_iter[i]
                        ),
                        "changes": [(-1, "N")],
                        "predictions": [baseline_prediction[i]],
                        "designed_sequence": "",
                    }
                )

        # do all possible mutations
        for i in range(n_sequences):
            mutagenesis[i] = generate_mutagenesis(
                sequence_onehot_prev_iter[i : i + 1],
                include_original=False,
                flanks=no_mutation_flanks,
            )
        mutagenesis_predictions = []
        for m in model:
            mutagenesis_prediction = m.predict(
                mutagenesis.reshape(
                    (n_sequences * TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ, seq_len, 4)
                ),
                verbose=False,
            )

            mutagenesis_prediction = mutagenesis_prediction.reshape(
                (
                    n_sequences,
                    TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ,
                    mutagenesis_prediction.shape[1],
                )
            )
            mutagenesis_predictions.append(mutagenesis_prediction)
        mutagenesis_predictions = np.mean(mutagenesis_predictions, axis=0)
        for i in range(n_sequences):
            best_mutation = enhancer_optimizer.get_best(
                mutated_predictions=mutagenesis_predictions[i],
                original_prediction=baseline_prediction[i],
                target=target,
                **kwargs,
            )
            sequence_onehot_prev_iter[i] = mutagenesis[
                i, best_mutation : best_mutation + 1, :
            ]
            if return_intermediate:
                mutation_index = best_mutation // 3 + no_mutation_flanks[0]
                changed_to = hot_encoding_to_sequence(
                    sequence_onehot_prev_iter[i, mutation_index, :]
                )
                intermediate_info_list[i]["changes"].append(
                    (mutation_index, changed_to)
                )
                intermediate_info_list[i]["predictions"].append(
                    mutagenesis_predictions[i][best_mutation]
                )

    # get final sequence
    for i in range(n_sequences):
        best_mutation = enhancer_optimizer.get_best(
            mutated_predictions=mutagenesis_predictions[i],
            original_prediction=baseline_prediction[i],
            target=target,
            **kwargs,
        )

        designed_sequence = hot_encoding_to_sequence(
            mutagenesis[i, best_mutation : best_mutation + 1, :]
        )

        designed_sequences.append(designed_sequence)

        if return_intermediate:
            intermediate_info_list[i]["designed_sequence"] = designed_sequence

    if return_intermediate:
        return intermediate_info_list, designed_sequences
    else:
        return designed_sequences


def motif_insertion(
    patterns: dict,
    model: keras.Model | list[keras.Model],
    target: int | np.ndarray,
    n_sequences: int = 1,
    insertions_per_pattern: dict | None = None,
    return_intermediate: bool = False,
    no_mutation_flanks: tuple[int, int] | None = None,
    target_len: int | None = None,
    preserve_inserted_motifs: bool = True,
    enhancer_optimizer: EnhancerOptimizer | None = None,
    starting_sequences: str | list | None = None,
    acgt_distribution: np.ndarray[float] | None = None,
    **kwargs: dict[str, Any],
) -> list | tuple[list[dict], list]:
    """
    Create synthetic enhancers using motif insertions.

    Parameters
    ----------
    patterns
        Dictionary of patterns to be implemented in the form {'pattern_name': 'pattern_sequence'}
    model
        A (list of) trained keras model(s) to design enhancers with.
        If a list of models is provided, the predictions will be averaged across all models.
    target
        Using the default weighted_difference optimization function this should be the index of the target class to design enhancers for.
        This gets passed to the `get_best` function of the EnhancerOptimizer, so can represent other target values too.
    n_sequences
        Number of enhancers to design.
    insertions_per_pattern
        Dictionary of number of patterns to be implemented in the form {'pattern_name': number_of_insertions}.
        If not provided, each pattern is inserted once.
    return_intermediate
        If True, returns a dictionary with predictions and changes made in intermediate steps.
    no_mutation_flanks
        A tuple specifying regions in each flank where no modifications should occur.
    target_len
        Length of the area in the center of the sequence to make insertions, ignored if `no_mutation_flanks` is set.
    preserve_inserted_motifs
        If True, prevents motifs from being inserted on top of previously inserted motifs.
    enhancer_optimizer
        An instance of EnhancerOptimizer, defining how sequences should be optimized.
        If None, a default EnhancerOptimizer will be initialized using `_weighted_difference`
        as optimization function.
    starting_sequences
        An optional DNA sequence or a list of DNA sequences that will be used instead of randomly generated
        sequences. If provided, n_sequences is ignored
    acgt_distribution
        An array of floats representing the distribution of A, C, G, and T in the genome (in that order).
        If the array is of shape (L, 4), it will be assumed to be per position. If it is of shape (4,), it will be assumed to be overall.
        If None, a uniform distribution will be used.
        This will be used to generate random sequences if starting_sequences is not provided.
        You can calculate these using :func:`~crested.utils.calculate_nucleotide_distribution`.
    kwargs
        Additional arguments passed to `get_best` function of EnhancerOptimizer.

    Returns
    -------
    A list of designed sequences, and if `return_intermediate=True`, a list of intermediate results.

    See Also
    --------
    crested.utils.calculate_nucleotide_distribution
    crested.tl.design.EnhancerOptimizer

    Examples
    --------
    >>> acgt_distribution = crested.utils.calculate_nucleotide_distribution(
    ...     my_anndata, genome, per_position=True
    ... )  # shape (L, 4)
    >>> target_idx = my_anndata.obs_names.index("my_celltype")
    >>> my_motifs = {
    ...     "motif1": "ACGTTTGA",
    ...     "motif2": "TGCA",
    ... }
    >>> (
    ...     intermediate_results,
    ...     designed_sequences,
    ... ) = crested.tl.design.motif_insertion(
    ...     patterns=my_motifs,
    ...     n_mutations=20,
    ...     target=target_idx,
    ...     model=my_trained_model,
    ...     n_sequences=1,
    ...     return_intermediate=True,
    ...     acgt_distribution=acgt_distribution,
    ... )
    """
    if enhancer_optimizer is None:
        enhancer_optimizer = EnhancerOptimizer(optimize_func=_weighted_difference)

    if not isinstance(model, list):
        model = [model]

    seq_len = model[0].input_shape[1]

    # Determine mutation flanks
    if no_mutation_flanks is not None and target_len is not None:
        logger.warning(
            "Both no_mutation_flanks and target_len set, using no_mutation_flanks."
        )
    elif no_mutation_flanks is None and target_len is not None:
        if (seq_len - target_len) % 2 == 0:
            no_mutation_flanks = ((seq_len - target_len) // 2,) * 2
        else:
            no_mutation_flanks = (
                (seq_len - target_len) // 2,
                (seq_len - target_len) // 2 + 1,
            )
    elif no_mutation_flanks is None and target_len is None:
        no_mutation_flanks = (0, 0)

    if insertions_per_pattern is None:
        insertions_per_pattern = dict.fromkeys(patterns, 1)

    # Generate initial sequences
    if starting_sequences is None:
        if acgt_distribution is None:
            logger.warning(
                "No nucleotide distribution provided. Using uniform distribution."
            )
        initial_sequences = create_random_sequences(
            n_sequences=n_sequences,
            seq_len=seq_len,
            acgt_distribution=acgt_distribution,
        )
    else:
        initial_sequences = parse_starting_sequences(starting_sequences)
        n_sequences = initial_sequences.shape[0]

    designed_sequences = []
    intermediate_info_list = []

    for idx, sequence in enumerate(initial_sequences):
        sequence_onehot = one_hot_encode_sequence(sequence)
        inserted_motif_locations = np.array([]) if preserve_inserted_motifs else None

        if return_intermediate:
            baseline_prediction = np.mean(
                [m.predict(sequence_onehot, verbose=False) for m in model], axis=0
            )
            intermediate_info_list.append(
                {
                    "initial_sequence": sequence,
                    "changes": [(-1, "N")],
                    "predictions": [baseline_prediction[0]],
                    "designed_sequence": "",
                }
            )

        # Insert motifs sequentially
        for pattern_name, num_insertions in insertions_per_pattern.items():
            motif_onehot = one_hot_encode_sequence(patterns[pattern_name])
            motif_length = motif_onehot.shape[1]

            for _ in range(num_insertions):
                baseline_prediction = np.mean(
                    [m.predict(sequence_onehot, verbose=False) for m in model], axis=0
                )

                # Generate all motif insertion possibilities
                mutagenesis, insertion_locations = generate_motif_insertions(
                    sequence_onehot,
                    motif_onehot,
                    flanks=no_mutation_flanks,
                    masked_locations=inserted_motif_locations,
                )

                # Predict changes
                mutagenesis_predictions = np.mean(
                    [m.predict(mutagenesis, verbose=False) for m in model], axis=0
                )

                # Select best insertion site
                best_mutation = enhancer_optimizer.get_best(
                    mutated_predictions=mutagenesis_predictions,
                    original_prediction=baseline_prediction,
                    target=target,
                    **kwargs,
                )

                sequence_onehot = mutagenesis[best_mutation : best_mutation + 1]

                if preserve_inserted_motifs:
                    inserted_motif_locations = np.append(
                        inserted_motif_locations,
                        [
                            insertion_locations[best_mutation] + i
                            for i in range(motif_length)
                        ],
                    )

                if return_intermediate:
                    insertion_index = insertion_locations[best_mutation]
                    intermediate_info_list[idx]["changes"].append(
                        (insertion_index, patterns[pattern_name])
                    )
                    intermediate_info_list[idx]["predictions"].append(
                        mutagenesis_predictions[best_mutation]
                    )

        designed_sequence = hot_encoding_to_sequence(sequence_onehot)
        designed_sequences.append(designed_sequence)

        if return_intermediate:
            intermediate_info_list[idx]["designed_sequence"] = designed_sequence

    return (
        (intermediate_info_list, designed_sequences)
        if return_intermediate
        else designed_sequences
    )
