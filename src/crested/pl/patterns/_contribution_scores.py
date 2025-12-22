"""Plot contribution scores."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise

from ._utils import (
    _plot_attribution_map,
    _plot_mutagenesis_map,
    grad_times_input_to_df,
    grad_times_input_to_df_mutagenesis,
    grad_times_input_to_df_mutagenesis_letters,
)


@log_and_raise(ValueError)
def _check_contrib_params(
    zoom_n_bases: int | None,
    scores: np.ndarray,
    class_labels: list | None,
    sequence_labels: list | None,
):
    """Check contribution scores parameters."""
    if zoom_n_bases is not None and zoom_n_bases > scores.shape[2]:
        raise ValueError(
            f"zoom_n_bases ({zoom_n_bases}) must be less than or equal to the number of bases in the sequence ({scores.shape[2]})"
        )
    if class_labels:
        if len(class_labels) != scores.shape[1]:
            raise ValueError(
                f"Number of class plot labels ({len(class_labels)}) must match the number of classes ({scores.shape[1]})."
            )
    if sequence_labels:
        if len(sequence_labels) != scores.shape[0]:
            raise ValueError(
                f"Number of sequence plot labels ({len(sequence_labels)}) must match the number of sequences ({scores.shape[0]})."
            )


def contribution_scores(
    scores: np.ndarray,
    seqs_one_hot: np.ndarray,
    sequence_labels: str | list | None = None,
    class_labels: str | list | None = None,
    zoom_n_bases: int | None = None,
    highlight_positions: list[tuple[int, int]] | None = None,
    highlight_kws: dict | None = None,
    method: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Visualize interpretation scores with optional highlighted positions.

    Contribution scores can be calculated using the :func:`~crested.tl.contribution_scores` method.

    Parameters
    ----------
    scores
        Contribution scores of shape (n_seqs, n_classes, n_bases, n_features).
    seqs_one_hot
        One-hot encoded corresponding sequences of shape (n_seqs, n_bases, n_features).
    sequence_labels
        Label or list of sequence labels (subplot titles) to add to the plot. Should have the same length as the number of sequences.
    class_labels
        Label or list of class labels to add to the plot. Should have the same length as the number of classes.
    zoom_n_bases
        Number of center bases to zoom in on. Default is None (no zooming).
    highlight_positions
        List of tuples with start and end positions to highlight. Default is None.
    highlight_kws
        Keywords to use for plotting highlights with :meth:`~matplotlib.axes.Axes.axvspan`.
        Default is {'edgecolor':  "red", 'facecolor': "none", 'linewidth': 0.5}
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default is 18.
    height
        Height of the newly created figure if `ax=None`. Default is 3.
    sharex
        Whether to share the x axes of the created plots. Default is False.
    sharey
        Whether to share the y axes of the created plots. Default is False; y limits are shared between sequences only instead.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `prediction`: `xlabel='Cell types'`, `ylabel='Prediction'`, `grid="y"`.

    See Also
    --------
    crested.pl.render_plot

    Examples
    --------
    >>> import numpy as np
    >>> scores = np.random.rand(1, 1, 100, 4)
    >>> seqs_one_hot = np.random.randint(0, 2, (1, 100, 4))
    >>> class_labels = "celltype_A"
    >>> sequence_labels = "chr1:100-200"
    >>> crested.pl.patterns.contribution_scores(
    ...     scores, seqs_one_hot, sequence_labels, class_labels
    ... )

    .. image:: ../../../../docs/_static/img/examples/patterns_contribution_scores.png
    """
    if isinstance(sequence_labels, str):
        sequence_labels = [sequence_labels]
    if isinstance(class_labels, str):
        class_labels = [class_labels]

    _check_contrib_params(zoom_n_bases, scores, class_labels, sequence_labels)

    if zoom_n_bases is None:
        zoom_n_bases = scores.shape[2]
    center = int(scores.shape[2] / 2)
    start_idx = center - int(zoom_n_bases / 2)
    scores = scores[:, :, start_idx : start_idx + zoom_n_bases, :]

    seq_length = scores.shape[2]
    total_classes = scores.shape[1]
    total_sequences = seqs_one_hot.shape[0]
    total_plots = total_sequences * total_classes

    # Set defaults
    plot_width = kwargs.pop('width') if 'width' in kwargs else seq_length // 10
    plot_height = kwargs.pop('height') if 'height' in kwargs else 2 * total_plots
    sharex = kwargs.pop('sharex') if 'sharex' in kwargs else False
    sharey = kwargs.pop('sharey') if 'sharey' in kwargs else False
    if "xlabel" not in kwargs and total_plots == 1:
        kwargs["xlabel"] = "Position"
    if "supxlabel" not in kwargs and total_plots > 1:
        kwargs["supxlabel"] = "Position"
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Scores"
    highlight_kws = {} if highlight_kws is None else highlight_kws.copy()
    if 'edgecolor' not in highlight_kws:
        highlight_kws['edgecolor'] = "red"
    if 'facecolor' not in highlight_kws:
        highlight_kws['facecolor'] = "none"
    if 'linewidth' not in highlight_kws:
        highlight_kws['linewidth'] = 0.5

    if total_plots > 1 and ax is not None:
        raise ValueError("Cannot provide a pre-existing axis if plotting more than one sequence/more than one class. Please only provide one sequence and one class, or don't provide `ax`.")

    fig, axs = create_plot(ax=ax, width=plot_width, height=plot_height, nrows=total_plots, sharex=sharex, sharey=sharey)

    if total_plots == 1:
        axs = [axs]

    plot_idx = 0
    for seq in range(total_sequences):
        seq_class_x = seqs_one_hot[seq, start_idx : start_idx + zoom_n_bases, :]

        if method == "mutagenesis":
            global_max = scores[seq].max() + 0.25 * np.abs(scores[seq].max())
            global_min = scores[seq].min() - 0.25 * np.abs(scores[seq].min())
        elif method == "mutagenesis_letters":
            mins = []
            maxs = []
            for i in range(total_classes):
                seq_class_scores = scores[seq, i, :, :]
                mins.append(
                    np.min(-np.sum(seq_class_scores * (1 - seq_class_x), axis=1) / 3)
                )
                maxs.append(
                    np.max(-np.sum(seq_class_scores * (1 - seq_class_x), axis=1) / 3)
                )
            global_max = np.array(maxs).max() + 0.25 * np.abs(np.array(maxs).max())
            global_min = np.array(mins).min() - 0.25 * np.abs(np.array(mins).min())
        else:
            mins = []
            maxs = []
            for i in range(total_classes):
                seq_class_scores = scores[seq, i, :, :]
                mins.append(np.min(seq_class_scores * seq_class_x))
                maxs.append(np.max(seq_class_scores * seq_class_x))
            global_max = np.array(maxs).max() + 0.25 * np.abs(np.array(maxs).max())
            global_min = np.array(mins).min() - 0.25 * np.abs(np.array(mins).min())

        for i in range(total_classes):
            seq_class_scores = scores[seq, i, :, :]
            ax = axs[plot_idx]
            plot_idx += 1

            if method == "mutagenesis":
                mutagenesis_df = grad_times_input_to_df_mutagenesis(
                    seq_class_x, seq_class_scores
                )
                _plot_mutagenesis_map(mutagenesis_df, ax=ax)
            elif method == "mutagenesis_letters":
                mutagenesis_df_letters = grad_times_input_to_df_mutagenesis_letters(
                    seq_class_x, seq_class_scores
                )
                _plot_attribution_map(mutagenesis_df_letters, ax=ax, return_ax=False)
            else:
                intgrad_df = grad_times_input_to_df(seq_class_x, seq_class_scores)
                _plot_attribution_map(intgrad_df, ax=ax, return_ax=False)

            ax.set_ylim([global_min, global_max])
            text_to_add = class_labels[i] if class_labels else None
            ax.annotate(text_to_add, (0.025, 0.7), xycoords= 'axes fraction', fontsize=16, ha="left", va="center")

            # Draw rectangles to highlight positions
            if highlight_positions:
                for start, end in highlight_positions:
                    ax.axvspan(
                        xmin=start-start_idx-0.5,
                        xmax=end-start_idx-0.5,
                        **highlight_kws
                    )

            ax.set_xticks(np.arange(0, zoom_n_bases, 50))

        # Set the title for the sequence (subplot)
        if sequence_labels:
            axs[plot_idx - total_classes].set_title(sequence_labels[seq], fontsize=14)

    return render_plot(fig, axs, **kwargs)
