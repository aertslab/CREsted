from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from crested.pl._utils import render_plot
from crested.utils._logging import log_and_raise

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


@log_and_raise(ValueError)
def _check_figure_grid_params(n_rows: int, n_cols: int, n_of_plots: int):
    """Check figure grid parameters."""
    if n_rows * n_cols < n_of_plots:
        raise ValueError(
            f"can't fit {n_of_plots} plots into {n_rows} rows and {n_cols} columns."
        )


@log_and_raise(ValueError)
def _check_target_classes(target_classes: list[str], obs_names: pd.Index | list[str]):
    """Check target classes."""
    for target in target_classes:
        if target not in obs_names:
            raise ValueError(
                f"target class {target} not in obs_names. All targets must be in obs_names."
            )


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

    _check_contrib_params(zoom_n_bases, scores_all[0], labels, None)

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


def enhancer_design_steps_predictions(
    intermediate: list[dict],
    target_classes: str | list[str],
    obs_names: pd.Index | list[str],
    seperate: bool = False,
    global_ylim: str = "minmax",
    n_rows: int | None = None,
    n_cols: int | None = None,
    alpha_seperate: float = 1.0,
    legend_seperate: bool = False,
    plot_color: str | tuple = (0.3, 0.5, 0.6),
    show_fliers: bool = False,
    fig_rescale: float = 1.0,
    **kwargs,
):
    """
    Visualize enhancer design prediction score progression.

    Parameters
    ----------
    intermediate
        Intermediate output from enhancer design when return_intermediate is True
    target_classes
        Target classes that the predictions will be plotted for. All target classes must be in obs_names.
    obs_names
        All class names either in the form of AnnData.obs_names or as a list.
    seperate
        Whether to plot each design enhancer seperately, or all together as a boxplot. Default is False.
    global_ylim
        Used to set global y-axis limits across plots. Can be one of 'classification', 'minmax' or None. Default is 'minmax'
        'classification' makes the y-axis limits (0, 1).
        'minmax' makes the y-axis limit minimum and maximum prediction across all the target classes of all designed enhancers
        If None, each plot has its y-axis limits seperately selected.
    n_rows
        Number of rows to use when more than one target class is selected.
    n_cols
        Number of columns to use when more than one target class is selected.
    alpha_seperate
        Line alpha for lines when seperate is True. Default is 1.0.
    legend_seperate
        Whether to plot a legend when seperate is True. Default is False.
    plot_color
        Boxplot color when seperate is False. Default is (0.3, 0.5, 0.6).
    show_fliers
        Whether to show fliers when seperate is False. Default is False.
    fig_rescale
        A scalar to scale the figure size up or down. Default is 1.0.
    """
    if not isinstance(obs_names, list):
        obs_names = list(obs_names)

    if isinstance(target_classes, str):
        target_classes = [target_classes]

    _check_target_classes(target_classes, obs_names)
    target_indexes = [obs_names.index(target_class) for target_class in target_classes]
    n_of_plots = len(target_classes)
    if n_rows is not None and n_cols is not None:
        _check_figure_grid_params(n_rows, n_cols, n_of_plots)
    elif n_rows is not None and n_cols is None:
        n_cols = n_of_plots // n_rows + (n_of_plots % n_rows > 0)
    elif n_rows is None and n_cols is not None:
        n_rows = n_of_plots // n_cols + (n_of_plots % n_cols > 0)
    elif n_rows is None and n_cols is None:
        if n_of_plots == 1:
            n_rows, n_cols = 1, 1
        elif n_of_plots == 2:
            n_rows, n_cols = 1, 2
        elif n_of_plots == 4:
            n_rows, n_cols = 2, 2
        else:
            n_cols = 3
            n_rows = n_of_plots // n_cols + (n_of_plots % n_cols > 0)

    max_prediction, min_prediction = 0, np.inf
    predictions_per_class = {}
    all_predictions = []
    for intermediate_dict in intermediate:
        design_predictions = np.zeros(
            (len(intermediate_dict["predictions"]), len(target_indexes))
        )
        for i, prediction in enumerate(intermediate_dict["predictions"]):
            design_predictions[i, :] = prediction[target_indexes]
        all_predictions.append(design_predictions)
        if np.max(design_predictions) > max_prediction:
            max_prediction = np.max(design_predictions)
        if np.min(design_predictions) < min_prediction:
            min_prediction = np.min(design_predictions)

    for idx in range(len(target_indexes)):
        predictions_per_class[target_classes[idx]] = np.column_stack(
            [pred_mat[:, idx] for pred_mat in all_predictions]
        )

    fig, ax = plt.subplots(n_rows, n_cols)
    for idx in range(n_rows * n_cols):
        if n_rows == 1 and n_cols == 1:
            curr_ax = ax
        elif n_rows == 1 or n_cols == 1:
            curr_ax = ax[idx]
        else:
            i, j = idx // n_cols, idx % n_cols
            curr_ax = ax[i, j]

        if idx >= len(target_indexes):
            ax[i, j].set_axis_off()
            continue
        else:
            target = target_classes[idx]

        if seperate:
            curr_ax.plot(
                predictions_per_class[target],
                marker="o",
                markersize=7,
                alpha=alpha_seperate,
                linewidth=0.5,
            )
            if legend_seperate:
                curr_ax.legend(range(len(intermediate)))
        else:
            curr_ax.boxplot(
                predictions_per_class[target].T,
                showfliers=show_fliers,
                capprops={"color": plot_color},
                boxprops={"color": plot_color},
                whiskerprops={"color": plot_color},
                flierprops={"markeredgecolor": plot_color},
                medianprops={"color": plot_color},
                meanprops={"color": plot_color},
            )
        if global_ylim == "classification":
            curr_ax.set_ylim(0, 1)
        elif global_ylim:
            curr_ax.set_ylim(0, max_prediction)
        curr_ax.set_title(f"Class {target}")
        curr_ax.set_xlabel("Steps")
        curr_ax.set_ylabel("Prediction Score")

    if "width" not in kwargs:
        kwargs["width"] = fig_rescale * 10 * n_cols
    if "height" not in kwargs:
        kwargs["height"] = fig_rescale * 10 * n_rows
    render_plot(fig, **kwargs)
