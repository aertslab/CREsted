"""Plot contribution scores."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from crested._logging import log_and_raise
from crested.pl._utils import render_plot

from ._utils import _plot_attribution_map, grad_times_input_to_df


@log_and_raise(ValueError)
def _check_contrib_params(
    zoom_n_bases: int | None,
    scores: np.ndarray,
    labels: list | None,
):
    """Check contribution scores parameters."""
    if zoom_n_bases is not None and zoom_n_bases > scores.shape[2]:
        raise ValueError(
            f"zoom_n_bases ({zoom_n_bases}) must be less than or equal to the number of bases in the sequence ({scores.shape[2]})"
        )
    if labels:
        if len(labels) != scores.shape[1]:
            raise ValueError(
                f"Number of plot labels ({len(labels)}) must match the number of classes ({scores.shape[1]}), since each class has a separate plot."
            )


def contribution_scores(
    scores: np.ndarray,
    seqs_one_hot: np.ndarray,
    labels: list | None = None,
    zoom_n_bases: int | None = None,
    highlight_positions: list[tuple[int, int]] | None = None,
    ylim: tuple | None = None,
    **kwargs,
):
    """
    Visualize interpretation scores with optional highlighted positions.

    Contribution scores can be calculated using the :func:`~crested.tl.Crested.calculate_contribution_scores` method.

    Parameters
    ----------
    scores
        Contribution scores of shape (n_seqs, n_classes, n_bases, n_features).
    seqs_one_hot
        One-hot encoded corresponding sequences of shape (n_seqs, n_bases, n_features).
    labels
        List of labels to add to the plot. Should have the same length as the number of classes.
    zoom_n_bases
        Number of center bases to zoom in on. Default is None (no zooming).
    highlight_positions
        List of tuples with start and end positions to highlight. Default is None.
    ylim
        Y-axis limits. Default is None.

    Examples
    --------
    >>> import numpy as np
    >>> scores = np.random.rand(1, 1, 100, 4)
    >>> seqs_one_hot = np.random.randint(0, 2, (1, 100, 4))
    >>> labels = ["class1"]
    >>> crested.pl.patterns.contribution_scores(scores, seqs_one_hot, labels)

    .. image:: ../../../../docs/_static/img/examples/contribution_scores.png
    """
    # Center and zoom
    _check_contrib_params(zoom_n_bases, scores, labels)

    if zoom_n_bases is None:
        zoom_n_bases = scores.shape[2]
    if labels and not isinstance(labels, list):
        labels = [str(labels)]
    center = int(scores.shape[2] / 2)
    start_idx = center - int(zoom_n_bases / 2)
    scores = scores[:, :, start_idx : start_idx + zoom_n_bases, :]

    global_min = scores.min()
    global_max = scores.max()

    # Plot
    logger.info(f"Plotting contribution scores for {seqs_one_hot.shape[0]} sequence(s)")
    for seq in range(seqs_one_hot.shape[0]):
        fig_height_per_class = 2
        fig = plt.figure(figsize=(50, fig_height_per_class * scores.shape[1]))
        for i in range(scores.shape[1]):
            seq_class_scores = scores[seq, i, :, :]
            seq_class_x = seqs_one_hot[seq, :, :]
            intgrad_df = grad_times_input_to_df(seq_class_x, seq_class_scores)
            ax = plt.subplot(scores.shape[1], 1, i + 1)
            _plot_attribution_map(intgrad_df, ax=ax, return_ax=False)
            if labels:
                class_name = labels[i]
            else:
                class_name = f"Class {i}"
            text_to_add = class_name
            if ylim:
                ax.set_ylim(ylim[0], ylim[1])
                x_pos = 5
                y_pos = 0.75 * ylim[1]
            else:
                ax.set_ylim([global_min, global_max])
                x_pos = 5
                y_pos = 0.75 * global_max
            ax.text(x_pos, y_pos, text_to_add, fontsize=16, ha="left", va="center")

            # Draw rectangles to highlight positions
            if highlight_positions:
                for start, end in highlight_positions:
                    ax.add_patch(
                        plt.Rectangle(
                            (
                                start - start_idx - 0.5,
                                global_min,
                            ),
                            end - start,
                            global_max - global_min,
                            edgecolor="red",
                            facecolor="none",
                            linewidth=0.5,
                        )
                    )

        plt.xlabel("Position")
        plt.xticks(np.arange(0, zoom_n_bases, 50))

        if "width" not in kwargs:
            kwargs["width"] = 50
        if "height" not in kwargs:
            kwargs["height"] = fig_height_per_class * scores.shape[1]
        if "xlabel" not in kwargs:
            kwargs["xlabel"] = "Position"
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = "Scores"

        render_plot(fig, **kwargs)
