from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from crested._logging import log_and_raise
from crested.pl._utils import render_plot

from ._contribution_scores import _check_contrib_params
from ._utils import (
    _plot_attribution_map,
    _plot_mutagenesis_map,
    grad_times_input_to_df,
    grad_times_input_to_df_mutagenesis,
)


@log_and_raise(ValueError)
def _check_ylim_params(global_ylim: int | None, ylim: np.ndarray):
    """Check contribution scores parameters."""
    if global_ylim is not None and global_ylim not in ["all", "per_design", "per_plot"]:
        raise ValueError("global_ylim must be one of all, per_design or per_plot")
    if ylim is not None and global_ylim is not None:
        logger.warning("Both ylim and global_ylim is set. Ignoring ylim.")


def _determine_min_max(
    scores_all,
    seqs_one_hot_all,
    center,
    zoom_n_bases,
    start_idx,
    method,
    class_idxs=None,
    designed_idxs=None,
):
    n_designs = len(scores_all)
    n_class = scores_all[0].shape[1]
    n_steps = scores_all[0].shape[0]

    if class_idxs is None:
        class_idxs = range(n_class)

    if designed_idxs is None:
        designed_idxs = range(n_designs)

    # construct loop list
    constructed_loop = []
    for des_idx in designed_idxs:
        for c_idx in class_idxs:
            for s_idx in range(n_steps):
                constructed_loop.append((s_idx, c_idx, des_idx))

    mins = []
    maxs = []

    for s_idx, c_idx, des_idx in constructed_loop:
        scores = scores_all[des_idx][:, :, start_idx : start_idx + zoom_n_bases, :]
        seq_class_scores = scores[s_idx, c_idx, :, :]
        if method == "mutagenesis":
            maxs.append(seq_class_scores.max() + 0.25 * np.abs(seq_class_scores.max()))
            mins.append(seq_class_scores.min() - 0.25 * np.abs(seq_class_scores.min()))
        else:
            seqs_one_hot = seqs_one_hot_all[des_idx]
            seq_class_x = seqs_one_hot[s_idx, start_idx : start_idx + zoom_n_bases, :]
            mins.append(np.min(seq_class_scores * seq_class_x))
            maxs.append(np.max(seq_class_scores * seq_class_x))

    if method == "mutagenesis":
        return min(mins), max(maxs)

    return np.array(mins).min() - 0.25 * np.abs(np.array(mins).min()), np.array(
        maxs
    ).max() + 0.25 * np.abs(np.array(maxs).max())


def enhancer_design_steps_contribution_scores(
    intermediate: list[dict],
    scores_all: np.ndarray,
    seqs_one_hot_all: np.ndarray,
    labels: list | None = None,
    zoom_n_bases: int | None = None,
    ylim: tuple | None = None,
    global_ylim: str | None = "per_plot",
    method: str | None = None,
    **kwargs,
):
    """
    Visualize enhancer design stepwise contribution scores.

    Contribution scores can be calculated using the :func:`~crested.tl.Crested.calculate_contribution_scores_enhancer_design` method.

    Parameters
    ----------
    intermediate
        Intermediate output from enhancer design when return_intermediate is True
    scores_all
        An array of a list of arrays of contribution scores of shape (n_seqs, n_classes, n_bases, n_features).
    seqs_one_hot_all
        An array of a list of arrays of one-hot encoded corresponding sequences of shape (n_seqs, n_bases, n_features).
    labels
        List of labels to add to the plot. Should have the same length as the number of classes.
    zoom_n_bases
        Number of center bases to zoom in on. Default is None (no zooming).
    ylim
        Y-axis limits, ignored if global_ylim is set. Default is None.
    global_ylim
        Used to set global y-axis limits across explanations. Can be one of 'all', 'per_design', 'per_plot' or None. Default is 'per_plot'
        'all' makes the y-axis limit same across all of the explanations.
        'per_design' makes the y-axis limit same across all of the steps and classes of a single designed sequence.
        'per_plot' makes y-axis limits same across all the steps but not the classes of a single designed sequence.
        If None, each explanation has its y-axis limits seperately selected.
    method
        Method used for calculating contribution scores. If mutagenesis, specify.
    """
    _check_ylim_params(global_ylim, ylim)

    if not isinstance(scores_all, list):
        scores_all = [scores_all]
    if not isinstance(seqs_one_hot_all, list):
        seqs_one_hot_all = [seqs_one_hot_all]

    _check_contrib_params(zoom_n_bases, scores_all[0], labels)

    # Center and zoom
    if zoom_n_bases is None:
        zoom_n_bases = scores_all[0].shape[2]
    if labels and not isinstance(labels, list):
        labels = [str(labels)]
    center = int(scores_all[0].shape[2] / 2)
    start_idx = center - int(zoom_n_bases / 2)

    if global_ylim == "all":
        if method == "mutagenesis":
            global_min, global_max = _determine_min_max(
                scores_all, seqs_one_hot_all, center, zoom_n_bases, start_idx, method
            )
        else:
            global_min, global_max = _determine_min_max(
                scores_all, seqs_one_hot_all, center, zoom_n_bases, start_idx, method
            )

    # Plot
    for intermediate_idx, (scores, seqs_one_hot) in enumerate(
        zip(scores_all, seqs_one_hot_all)
    ):
        intermediate_current = intermediate[intermediate_idx]
        if global_ylim == "per_design":
            if method == "mutagenesis":
                global_min, global_max = _determine_min_max(
                    scores_all,
                    seqs_one_hot_all,
                    center,
                    zoom_n_bases,
                    start_idx,
                    method,
                    designed_idxs=[intermediate_idx],
                )
            else:
                global_min, global_max = _determine_min_max(
                    scores_all,
                    seqs_one_hot_all,
                    center,
                    zoom_n_bases,
                    start_idx,
                    method,
                    designed_idxs=[intermediate_idx],
                )

        for class_idx in range(scores.shape[1]):
            logger.info(
                f"Plotting contribution scores for {seqs_one_hot.shape[0]} sequence(s)"
            )
            number_of_steps = seqs_one_hot.shape[0]
            fig, ax = plt.subplots(
                number_of_steps, 1, figsize=(50, 2 * number_of_steps)
            )
            if global_ylim == "per_plot":
                if method == "mutagenesis":
                    global_min, global_max = _determine_min_max(
                        scores_all,
                        seqs_one_hot_all,
                        center,
                        zoom_n_bases,
                        start_idx,
                        method,
                        designed_idxs=[intermediate_idx],
                        class_idxs=[class_idx],
                    )
                else:
                    global_min, global_max = _determine_min_max(
                        scores_all,
                        seqs_one_hot_all,
                        center,
                        zoom_n_bases,
                        start_idx,
                        method,
                        designed_idxs=[intermediate_idx],
                        class_idxs=[class_idx],
                    )

            for seq in range(seqs_one_hot.shape[0]):
                seq_class_x = seqs_one_hot[seq, start_idx : start_idx + zoom_n_bases, :]

                seq_class_scores = scores[seq, class_idx, :, :]
                if method == "mutagenesis":
                    mutagenesis_df = grad_times_input_to_df_mutagenesis(
                        seq_class_x, seq_class_scores
                    )
                    _plot_mutagenesis_map(mutagenesis_df, ax=ax[seq])
                else:
                    intgrad_df = grad_times_input_to_df(seq_class_x, seq_class_scores)
                    _plot_attribution_map(intgrad_df, ax=ax[seq], return_ax=False)

                if global_ylim in ["all", "per_design", "per_plot"]:
                    ax[seq].set_ylim([global_min, global_max])
                elif ylim:
                    ax[seq].set_ylim(ylim[0], ylim[1])
                elif global_ylim is None:
                    current_ylim = ax[seq].get_ylim()
                    ax[seq].set_ylim((current_ylim[0] * 1.1, current_ylim[1] * 1.1))

                current_ylim = ax[seq].get_ylim()

                # Set change step as title
                if seq == 0:
                    current_mut_title = "Random Sequence"
                else:
                    current_mut_title = f"Step {seq}"
                ax[seq].set_title(current_mut_title)

                # Draw rectangles to highlight positions
                change_loc, change = intermediate_current["changes"][seq]
                if change_loc != -1:
                    start, end = change_loc, change_loc + len(change)
                    ax[seq].add_patch(
                        plt.Rectangle(
                            (
                                start - start_idx - 0.5,
                                current_ylim[0],
                            ),
                            end - start,
                            current_ylim[1] - current_ylim[0],
                            edgecolor="red",
                            facecolor="none",
                            linewidth=0.5,
                        )
                    )
            if labels:
                class_name = labels[class_idx]
            else:
                class_name = f"Class {class_idx}"
            kwargs["title"] = class_name
            plt.xticks(np.arange(0, zoom_n_bases, 50))

            if "width" not in kwargs:
                kwargs["width"] = 50
            if "height" not in kwargs:
                kwargs["height"] = 2 * number_of_steps
            if "supxlabel" not in kwargs:
                kwargs["supxlabel"] = "Position"
            if "supylabel" not in kwargs:
                kwargs["supylabel"] = "Scores"
            if "tight_rect" not in kwargs:
                kwargs["tight_rect"] = (0.02, 0.02, 1, 0.98)
            render_plot(fig, **kwargs)


def enhancer_design_steps_predictions():
    # TODO implement
    pass
