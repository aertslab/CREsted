"""Plot contribution scores."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import logomaker
import matplotlib.pyplot as plt
import numpy as np

from crested.pl._utils import create_plot, render_plot
from crested.utils import hot_encoding_to_sequence
from crested.utils._logging import log_and_raise

from ._utils import (
    _plot_attribution_map,
    _plot_mutagenesis_map,
    _process_gradients,
    _process_mutagenesis,
    _process_mutagenesis_letters,
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
    highlight_positions: tuple[int, int] | list[tuple[int, int]] | None = None,
    method: Literal['mutagenesis', 'mutagenesis_letters'] | None = None,
    plot_kws: dict | None = None,
    highlight_kws: dict | None = None,
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
    method
        Default is None (for gradient-based contributions). If plotting mutagenesis values, set to `'mutagenesis_letters'`
        (to visualize average effects as letters) or `mutagenesis` (to visualize in a legacy way).
    plot_kws
        Extra keyword arguments passed to the underlying plotting function.
        If `method` is `None` or `'mutagenesis_letters'`, passed to  `_plot_attribution_map` and on to `logomaker.Logo`.
        If `method` is `'mutagenesis'`, passed to `_plot_mutagenesis_map` and on to :meth:`~matplotlib.axes.Axes.bar`.
    highlight_kws
        Keywords to use for plotting highlights with :meth:`~matplotlib.axes.Axes.axvspan`.
        Default is {'edgecolor':  "red", 'facecolor': "none", 'linewidth': 0.5}
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default is n_bases//10.
    height
        Height of the newly created figure if `ax=None`. Default is 2*n_seqs*n_classes.
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
    >>> scores = np.random.random_sample((1, 1, 100, 4))
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

    # Add extra sequence/class/etc dims if inputs are not quite properly shaped
    if seqs_one_hot.ndim == 2:
        seqs_one_hot = np.expand_dims(seqs_one_hot, 0)
    if scores.ndim == 2:
        scores = np.expand_dims(scores, (0, 1))
    elif scores.ndim == 3:
        if scores.shape[0] == seqs_one_hot.shape[0]:
            scores = np.expand_dims(scores, 1)
        else:
            scores = np.expand_dims(scores, 0)

    if highlight_positions is not None:
        if not isinstance(highlight_positions[0], Sequence):
            highlight_positions = [highlight_positions]

    _check_contrib_params(zoom_n_bases, scores, class_labels, sequence_labels)

    if zoom_n_bases is None:
        zoom_n_bases = scores.shape[2]
    center = int(scores.shape[2] / 2)
    start_idx = center - int(zoom_n_bases / 2)
    scores = scores[:, :, start_idx:start_idx+zoom_n_bases, :]
    seqs_one_hot = seqs_one_hot[:, start_idx:start_idx+zoom_n_bases, :]

    seq_length = scores.shape[2]
    total_classes = scores.shape[1]
    total_sequences = seqs_one_hot.shape[0]
    total_plots = total_sequences * total_classes

    # Set defaults
    if "xlabel" not in kwargs and total_plots == 1:
        kwargs["xlabel"] = "Position"
    if "supxlabel" not in kwargs and total_plots > 1:
        kwargs["supxlabel"] = "Position"
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Scores"
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    highlight_kws = {} if highlight_kws is None else highlight_kws.copy()
    if 'edgecolor' not in highlight_kws:
        highlight_kws['edgecolor'] = "red"
    if 'facecolor' not in highlight_kws:
        highlight_kws['facecolor'] = "none"
    if 'linewidth' not in highlight_kws:
        highlight_kws['linewidth'] = 0.5

    if total_plots > 1 and ax is not None:
        raise ValueError("Cannot provide a pre-existing axis if plotting more than one sequence/more than one class. Please only provide one sequence and one class, or don't provide `ax`.")

    sharing_y = 'sharey' in kwargs and kwargs['sharey']
    fig, axs = create_plot(
        ax=ax,
        kwargs_dict=kwargs,
        default_width=seq_length//10,
        default_height=2*total_plots,
        nrows=total_plots,
        default_sharex=False,
        default_sharey=False
    )

    if total_plots == 1:
        axs = [axs]

    plot_idx = 0
    for seq_i in range(total_sequences):
        # Gather this sequence's seq and score values
        seq_x = seqs_one_hot[seq_i, ...]
        seq_x_str = hot_encoding_to_sequence(seq_x)
        seq_scores_raw = scores[seq_i, ...]

        # Process values depending on method
        if method == 'mutagenesis':
            # Set reference nucleotides to nan to only plot alternative nucleotides
            seq_scores = _process_mutagenesis(seq=seq_x, scores=seq_scores_raw)
        elif method == 'mutagenesis_letters':
            # Keep only values of non-reference values and take mean of those 3
            seq_scores = _process_mutagenesis_letters(seq=seq_x, scores=seq_scores_raw)
        else:
            # Gradients: keep only scores for the reference nucleotide
            seq_scores = _process_gradients(seq=seq_x, scores=seq_scores_raw)

        # Get min and max ylims across all classes
        sequence_min = np.nanmin(seq_scores) - 0.25*np.abs(np.nanmin(seq_scores))
        sequence_max = np.nanmax(seq_scores) + 0.25*np.abs(np.nanmax(seq_scores))

        for class_i in range(total_classes):
            ax = axs[plot_idx]
            plot_idx += 1

            # Plot values for this sequence x class combo
            if method == "mutagenesis":
                _plot_mutagenesis_map(seq_scores[class_i], ax=ax, **plot_kws)
            else:
                logomaker_df = logomaker.saliency_to_matrix(seq=seq_x_str, values=seq_scores[class_i])
                _plot_attribution_map(logomaker_df, ax=ax, return_ax=False, **plot_kws)

            # Handle layout
            if not sharing_y:
                ax.set_ylim([sequence_min, sequence_max])
            if class_labels is not None:
                # Plot at bottom half if mutagenesis scatter (usually negative values), top half for letters (usually positive)
                label_rel_y = 0.3 if method == 'mutagenesis' else 0.7
                ax.annotate(class_labels[class_i], (0.025, label_rel_y), xycoords= 'axes fraction', fontsize=16, ha="left", va="center")

            # Draw highlights
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
            axs[plot_idx - total_classes].set_title(sequence_labels[seq_i], fontsize=14)

    return render_plot(fig, axs, **kwargs)
