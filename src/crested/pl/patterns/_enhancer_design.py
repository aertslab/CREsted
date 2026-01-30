from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise

from ._contribution_scores import contribution_scores


@log_and_raise(ValueError)
def _check_ylim_params(global_ylim: int | None, ylim: np.ndarray):
    """Check contribution scores parameters."""
    if global_ylim is not None and global_ylim not in ["all", "per_design", "per_plot"]:
        raise ValueError("global_ylim must be one of 'all', 'per_design' or 'per_plot' or None.")
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

def enhancer_design_steps_contribution_scores(
    intermediate: list[dict],
    scores_all: list[np.ndarray] | np.ndarray,
    seqs_one_hot_all: list[np.ndarray] | np.ndarray,
    sequence_labels: list | None = None,
    class_labels: list | None = None,
    zoom_n_bases: int | None = None,
    ylim: tuple[float, float] | list[tuple[float, float]] | None = None,
    global_ylim: Literal["all", "per_design", "per_plot"] | None = "per_plot",
    method: Literal["mutagenesis", "mutagenesis_letters"] | None = None,
    highlight_kws: dict | None = None,
    show: bool = True,
    labels: str = 'deprecated',
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]] | list[tuple[plt.Figure, list[plt.Axes]]]| None:
    """
    Visualize enhancer design stepwise contribution scores.

    Contribution scores can be calculated using the :func:`~crested.tl.contribution_scores` method.

    Parameters
    ----------
    intermediate
        Intermediate output from enhancer design when return_intermediate is True
    scores_all
        A list of contribution scores arrays for each designed sequence, of shape [(seq_steps, n_classes, n_bases, n_features), ...], like from :func:`~crested.tl.contribution_scores`.
    seqs_one_hot_all
        A list of one-hot encoded corresponding sequence arrays of shape [(seq_steps, n_bases, n_features), ...], like from :func:`~crested.utils.derive_intermediate_sequences()`.
    sequence_labels
        List of sequence labels ot add to the plot. Should have the same length as the number of designed sequences.
    class_labels
        List of class labels to add to the plot. Should have the same length as the number of classes.
    zoom_n_bases
        Number of center bases to zoom in on. Default is None (no zooming).
    ylim
        Y-axis limits, ignored if global_ylim is set. Can be a single tuple or a tuple for each designed sequence. Default is None.
    global_ylim
        Used to set global y-axis limits across explanations. Can be one of 'all', 'per_design', 'per_plot' or None. Default is 'per_plot'
        'all' makes the y-axis limit same across all of the explanations.
        'per_design' makes the y-axis limit same across all of the steps and classes of a single designed sequence.
        'per_plot' makes y-axis limits same across all the steps but not the classes of a single designed sequence.
        If None, each explanation has its y-axis limits separately selected.
    method
        Default is None (for gradient-based contributions). If plotting mutagenesis values, set to `'mutagenesis_letters'`
        (to visualize average effects as letters) or `mutagenesis` (to visualize in a legacy way).
    highlight_kws
        Keywords to use for plotting changed basepairs with :meth:`~matplotlib.axes.Axes.axvspan`.
        Default is {'edgecolor':  "red", 'facecolor': "none", 'linewidth' :0.5}
    show
        Whether to show all plots or return the (list of) figure and axes instead.
    width
        Width of each created figure. Default is 50.
    height
        Height of each created figure. Default is 2*`n_seqs`.
    sharex
        Whether to share the x axes of the created subplots within each figure. Default is False.
    sharey
        Whether to share the y axes of the created subplots within each figure. Default is False.
    kwargs
        Additional arguments passed to :func:`~crested.pl.patterns.contribution_scores` to control contribution score settings and on to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.patterns.contribution_scores` and :func:`~crested.pl.render_plot` for details.
        Custom defaults for `enhancer_design_steps_contribution_scores`: `suptitle_fontsize=26`, `tight_rect=[0, 0, 1, 0.98]`.

    Returns
    -------
    If `show=False`, a (fig, axs) tuple (if plotting one sequence and one class), or a list of (fig, axs) tuples (if plotting multiple sequences and/or classes).

    See Also
    --------
    crested.pl.render_plot
    crested.pl.patterns.contribution_scores

    Example
    --------
    >>> crested.pl.patterns.enhancer_design_steps_contribution_scores(
    ...     intermediate_results,
    ...     scores,
    ...     one_hot_encoded_sequences,
    ...     labels=["L5ET"],
    ...     highlight_kws={'facecolor': 'green', 'edgecolor': 'green', 'alpha': 0.1},
    ... )

    .. image:: ../../../../docs/_static/img/examples/patterns_enhancer_design_steps_contribution_scores.png
    """
    if labels != 'deprecated':
        class_labels = labels
        logger.warning("Argument `labels` is renamed; please use `class_labels` instead.")

    _check_ylim_params(global_ylim, ylim)

    # Handle defaults
    if 'suptitle_fontsize' not in kwargs:
        kwargs['suptitle_fontsize'] = 26
    if 'tight_rect' not in kwargs:
        kwargs['tight_rect'] = [0, 0, 1, 0.97]
    highlight_kws = {} if highlight_kws is None else highlight_kws.copy()
    if 'edgecolor' not in highlight_kws:
        highlight_kws['edgecolor'] = "red"
    if 'facecolor' not in highlight_kws:
        highlight_kws['facecolor'] = "none"
    if 'linewidth' not in highlight_kws:
        highlight_kws['linewidth'] = 0.5

    if not isinstance(scores_all, list):
        scores_all = [scores_all]
    if not isinstance(seqs_one_hot_all, list):
        seqs_one_hot_all = [seqs_one_hot_all]
    if ylim is not None and not isinstance(ylim[0], Sequence):
        ylim = [ylim]*len(scores_all)

    # Check inputs
    assert len(scores_all) == len(seqs_one_hot_all), f"Number of entries in the scores list ({len(scores_all)}) should be the same as the number of entries in the sequence list ({len(seqs_one_hot_all)})."
    if class_labels is not None:
        assert all(s.shape[1] == len(class_labels) for s in scores_all), f"Number of class_labels ({len(class_labels)}) should match number of classes, but found {[s.shape[1] for s in scores_all]} classes."
    if sequence_labels is not None:
        assert len(seqs_one_hot_all) == len(sequence_labels), f"Number of sequence_labels ({len(sequence_labels)}) should match number of designed sequences ({len(seqs_one_hot_all)})."

    if global_ylim is None:
        sharey = False
    else:
        sharey = True
        ylim = None

    # Goal of plotting: make one plot of n steps, for each designed seq x class combination.
    return_list = []
    design_idx_per_plot = []
    ylim_per_design = []
    for design_idx in range(len(scores_all)):
        ymin_sublist = []
        ymax_sublist = []
        n_steps, n_classes, seq_len, _ = scores_all[design_idx].shape
        step_labels = ["Random sequence"] + [f"Step {i}" for i in range(1, n_steps)]
        for class_idx in range(n_classes):
            # Plot intermediate sequences and scores for this designed sequence x class combo
            seq_label = f"Sequence {design_idx}" if sequence_labels is None else sequence_labels[design_idx]
            class_label = f"Class {class_idx}" if class_labels is None else class_labels[class_idx]
            fig, axs = contribution_scores(
                scores=scores_all[design_idx][:, class_idx, ...],
                seqs_one_hot=seqs_one_hot_all[design_idx],
                sequence_labels=step_labels, # Sequence labels per step
                class_labels=None,
                zoom_n_bases=zoom_n_bases,
                method=method,
                sharey=sharey,
                ylim=ylim if ylim is not None else None, # TODO: check this
                suptitle=f"{seq_label} - {class_label}",
                show=False,
                **kwargs,
            )
            # Draw highlights
            start_idx = int(seq_len / 2) - int(zoom_n_bases / 2) if zoom_n_bases is not None else 0 # Get rel shift if zoom_n_bases is used
            for step_idx in range(n_steps):
                change_loc, change = intermediate[design_idx]["changes"][step_idx]
                if change_loc != -1:
                    start, end = change_loc, change_loc+len(change)
                    axs[step_idx].axvspan(
                        xmin=start-start_idx-0.5,
                        xmax=end-start_idx-0.5,
                        **highlight_kws
                    )
            return_list.append((fig, axs))
            ymin_sublist.append(min(ax.get_ylim()[0] for ax in axs))
            ymax_sublist.append(max(ax.get_ylim()[1] for ax in axs))
            design_idx_per_plot.append(design_idx)
        # Get furthest ylims per design
        ylim_per_design.append((min(ymin_sublist), max(ymax_sublist)))

    # Handle global_ylim
    if global_ylim == 'all':
        total_ylim = (min(ylim[0] for ylim in ylim_per_design),  max(ylim[1] for ylim in ylim_per_design))

    for i, (_, axs) in enumerate(return_list):
        design_idx = design_idx_per_plot[i]
        for ax in axs:
            if global_ylim == 'all':
                ax.set_ylim(total_ylim)
            elif global_ylim == 'per_design':
                ax.set_ylim(ylim_per_design[design_idx])

    if show:
        plt.show()
    else:
        return return_list[0] if len(return_list) == 1 else return_list

def enhancer_design_steps_predictions(
    intermediate: list[dict],
    target_classes: str | list[str],
    obs_names: pd.Index | list[str],
    separate: bool = False,
    n_rows: int | None = None,
    n_cols: int | None = None,
    legend_separate: bool = False,
    plot_color: str | tuple = (0.3, 0.5, 0.6),
    fig_rescale: float = 1.0,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, list[plt.Axes]] | None:
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
    separate
        Whether to plot each designed enhancer separately, or all together as a boxplot. Default is False.
    n_rows
        Number of rows to use when more than one target class is selected.
        If None, will infer from `n_cols`. If both are None, creates a square grid.
    n_cols
        Number of columns to use when more than one target class is selected.
        If None, will infer from `n_rows`. If both are None, creates a square grid.
    legend_separate
        Whether to plot a legend when separate is True. Default is False.
    plot_color
        Boxplot color when separate is False. Default is (0.3, 0.5, 0.6).
    fig_rescale
        A scalar to scale the figure size up or down. Default is 1.0.
    plot_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.plot` (if `separate=True`)/:meth:`~matplotlib.axes.Axes.boxplot` (if `separate=False`).
        Defaults:
        `separate=True`: `{'marker': "o",'markersize': 7, 'linewidth': 0.5}`
        `separate=False`: `{"showfliers": False, "capprops"/"boxprops"/"whiskerprops"/"flierprops"/"medianprops"/"meanprops": {"color": plot_color}}`
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if not supplying `ax`. Default is `fig_rescale`*10*`n_rows`.
    height
        Height of the newly created figure if not supplying `ax`. Default is `fig_rescale`*10*`n_cols`.
    sharex
        Whether to share the x axes of the created plots. Default is False.
    sharey
        Whether to share the y axes of the created plots. Default is True.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.

    See Also
    --------
    crested.pl.render_plot

    Example
    --------
    >>> crested.pl.patterns.enhancer_design_steps_predictions(
    ...     intermediate_results,
    ...     target_classes="L5ET",
    ...     obs_names=adata.obs_names,
    ...     separate=True,
    ... )

    .. image:: ../../../../docs/_static/img/examples/patterns_enhancer_design_steps_predictions.png
    """
    if 'seperate' in kwargs:
        separate = kwargs.pop('seperate')
        logger.warning("Please use argument `separate` instead of `seperate`.")
    if 'legend_seperate' in kwargs:
        legend_separate = kwargs.pop('legend_seperate')
        logger.warning("Please use argument `legend_separate` instead of `legend_seperate`.")

    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'alpha_seperate' in kwargs:
        logger.warning(f"Please use argument `plot_kws={{'alpha': {kwargs['alpha_seperate']}}}` instead of `alpha_seperate`.")
        plot_kws['alpha'] = kwargs['alpha_seperate']
    if 'show_fliers' in kwargs:
        logger.warning(f"Please use argument `plot_kws={{'showfliers': {kwargs['show_fliers']}}}` instead of `show_fliers`.")
        plot_kws['showfliers'] = kwargs['show_fliers']
    if 'global_ylim' in kwargs:
        if kwargs['global_ylim'] == 'minmax':
            logger.warning("Argument `global_ylim` is superseded by `ylim` and `sharey`. Please set `sharey=True` instead of `global_ylim='minmax'`.")
            kwargs['sharey'] = True
        elif kwargs['global_ylim'] == 'classification':
            logger.warning("Argument `global_ylim` is superseded by `ylim` and `sharey`. Please set `ylim=(0,1) instead of `global_ylim='classification'`.")
            kwargs['ylim'] = (0, 1)
        del kwargs['global_ylim']

    if not isinstance(obs_names, list):
        obs_names = list(obs_names)

    if isinstance(target_classes, str):
        target_classes = [target_classes]

    _check_target_classes(target_classes, obs_names)
    if len(target_classes) > 1 and ax is not None:
        raise ValueError("A pre-existing axis can only be used if plotting a single target class.")

    n_of_plots = len(target_classes)
    target_indexes = [obs_names.index(target_class) for target_class in target_classes]
    if n_rows is not None and n_cols is not None:
        _check_figure_grid_params(n_rows, n_cols, n_of_plots)
    elif n_rows is not None and n_cols is None:
        n_cols = n_of_plots // n_rows + (n_of_plots % n_rows > 0)
    elif n_rows is None and n_cols is not None:
        n_rows = n_of_plots // n_cols + (n_of_plots % n_cols > 0)
    elif n_rows is None and n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_of_plots)))
        n_rows = int(np.ceil(n_of_plots / n_cols))

    predictions_per_class = {}
    all_predictions = []
    for intermediate_dict in intermediate:
        design_predictions = np.zeros(
            (len(intermediate_dict["predictions"]), len(target_indexes))
        )
        for i, prediction in enumerate(intermediate_dict["predictions"]):
            design_predictions[i, :] = prediction[target_indexes]
        all_predictions.append(design_predictions)

    for idx in range(len(target_indexes)):
        predictions_per_class[target_classes[idx]] = np.column_stack(
            [pred_mat[:, idx] for pred_mat in all_predictions]
        )

    # Set defaults
    if 'title' not in kwargs:
        kwargs['title'] = [f"Class {target}" for target in target_classes]
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = "Steps"
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = "Prediction score"
    if 'grid' not in kwargs:
        kwargs['grid'] = 'y'
    if 'ylim' not in kwargs:
        kwargs['ylim'] = (0, None)
    fig, axs = create_plot(
        ax=ax,
        kwargs_dict=kwargs,
        default_width=fig_rescale*10*n_cols,
        default_height=fig_rescale*10*n_rows,
        nrows=n_rows,
        ncols=n_cols,
        default_sharex=False,
        default_sharey=True,
        squeeze=False
    )
    separate_plot_kws = {
        'marker' : "o",
        'markersize': 7,
        'linewidth': 0.5
    }
    separate_plot_kws.update(plot_kws)
    box_plot_kws = {
        "showfliers": False,
        "capprops": {"color": plot_color},
        "boxprops": {"color": plot_color},
        "whiskerprops": {"color": plot_color},
        "flierprops": {"markeredgecolor": plot_color},
        "medianprops": {"color": plot_color},
        "meanprops": {"color": plot_color}
    }
    box_plot_kws.update(plot_kws)

    for idx in range(n_rows * n_cols):
        i, j = idx // n_cols, idx % n_cols

        if idx >= len(target_indexes):
            axs[i, j].set_axis_off()
            continue
        else:
            target = target_classes[idx]

        if separate:
            axs[i, j].plot(
                predictions_per_class[target],
                **separate_plot_kws
            )
            if legend_separate:
                axs[i, j].legend(range(len(intermediate)))
        else:
            axs[i, j].boxplot(
                predictions_per_class[target].T,
                **box_plot_kws
            )

    return render_plot(fig, axs, **kwargs)
