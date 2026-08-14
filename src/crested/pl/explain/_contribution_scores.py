"""Plot contribution scores."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from crested.pl._utils import _parse_coordinates_input, create_plot, render_plot
from crested.utils._logging import log_and_raise

from ._utils import (
    _build_attribution_logo,
    _draw_logo,
    _plot_mutagenesis_map,
    _process_gradients,
    _process_mutagenesis,
    _process_mutagenesis_letters,
)


def contribution_scores(
    scores: np.ndarray,
    seqs_one_hot: np.ndarray,
    sequence_labels: str | list[str] | None = None,
    class_labels: str | list[str] | None = None,
    zoom_n_bases: int | None = None,
    coordinates: str | tuple | list[str] | list[tuple] | None = None,
    highlight_positions: tuple[int, int] | list[tuple[int, int]] | None = None,
    x_shift: int = 0,
    method: Literal['mutagenesis', 'mutagenesis_letters'] | None = None,
    plot_kws: dict | None = None,
    highlight_kws: dict | None = None,
    sharey: Literal["sequence", True, False] = "sequence",
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
    coordinates
        Optional, a (list of) string or tuple of coordinates that are being plotted between per sequence, to set the x coordinates.
        Every entry can be a parsable chr:start-region(:strand) string, or a tuple with ((chr), start, end, (strand)), with chr and strand being optional.
        If strand is provided and '-', runs the x coordinates from end to start, as expected with a negative strand region.
    highlight_positions
        List of tuples with start and end positions to highlight. Default is None.
        Positions are within the full sequence length before zooming, or optionally genomic values if using `coordinates`.
        Like indexing, highlights take [start, end) (aka start to end-1).
    x_shift
        Number of base pairs to shift left or right for visualizing specific subsets of the region. Only use when combined with zooming in. Default is zero.
    method
        Default is None (for gradient-based contributions). If plotting mutagenesis values, set to `'mutagenesis_letters'`
        (to visualize average effects as letters) or `mutagenesis` (to visualize in a legacy way).
    plot_kws
        Extra keyword arguments passed to the underlying plotting function.
        If `method` is `None` or `'mutagenesis_letters'`, passed to `_build_attribution_logo` and on to `fast_logomaker.FastLogo`.
        If `method` is `'mutagenesis'`, passed to `_plot_mutagenesis_map` and on to :meth:`~matplotlib.axes.Axes.bar`.
    highlight_kws
        Keywords to use for plotting highlights with :meth:`~matplotlib.axes.Axes.axvspan`.
        Default is {'edgecolor':  "red", 'facecolor': "none", 'linewidth': 0.5}
    sharey
        Whether to share the y axes of the created plots. Default is 'sequence', which shared between classes for one sequence but not between sequences.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default is n_bases//10.
    height
        Height of the newly created figure if `ax=None`. Default is 0.5+2.25*n_seqs*n_classes.
    sharex
        Whether to share the x axes of the created plots. Default is False.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `contribution_scores`: `ylabel='Scores'`, `h_pad=0.2` (for :func:`~crested.pl.create_plot`).

    See Also
    --------
    crested.pl.render_plot
    crested.tl.contribution_scores

    Examples
    --------
    >>> import numpy as np
    >>> scores = np.random.exponential(10, (1, 1, 200, 4))
    >>> seqs_one_hot = np.eye(4)[None, np.random.randint(4, size=200)]
    >>> class_of_interest = "celltype_A"
    >>> region_of_interest = "chr1:100-300"
    >>> crested.pl.explain.contribution_scores(
    ...     scores=scores,
    ...     seqs_one_hot=seqs_one_hot,
    ...     sequence_labels=region_of_interest,
    ...     class_labels=class_of_interest,
    ...     coordinates=region_of_interest
    ... )

    .. image:: /_static/img/examples/explain_contribution_scores.png
    """

    @log_and_raise(ValueError)
    def _check_contrib_params():
        """Check contribution scores parameters."""
        if zoom_n_bases is not None and zoom_n_bases > scores.shape[2]:
            raise ValueError(
                f"zoom_n_bases ({zoom_n_bases}) must be less than or equal to the number of bases in the sequence ({scores.shape[2]})"
            )
        if class_labels is not None:
            if len(class_labels) != scores.shape[1]:
                raise ValueError(
                    f"Number of class plot labels ({len(class_labels)}) must match the number of classes ({scores.shape[1]})."
                )
        if sequence_labels is not None:
            if len(sequence_labels) != scores.shape[0]:
                raise ValueError(
                    f"Number of sequence plot labels ({len(sequence_labels)}) must match the number of sequences ({scores.shape[0]})."
                )
        if sharey not in [True, False, "sequence"]:
            raise ValueError(f"`sharey` must be True, False or 'sequence', not {sharey}")

    if isinstance(sequence_labels, str):
        sequence_labels = [sequence_labels]
    if isinstance(class_labels, str):
        class_labels = [class_labels]
    if isinstance(coordinates, tuple) or isinstance(coordinates, str):
        coordinates = [coordinates]
    if highlight_positions is not None:
        if not isinstance(highlight_positions[0], Sequence):
            highlight_positions = [highlight_positions]

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

    _check_contrib_params()

    if zoom_n_bases is None:
        zoom_n_bases = scores.shape[2]
    center = int(scores.shape[2] / 2)
    start_idx = center - int(zoom_n_bases / 2) + x_shift
    if start_idx < 0 or (start_idx + zoom_n_bases > scores.shape[2]):
        raise ValueError(
            f"Parameter x_shift={x_shift} with zoom={zoom_n_bases} "
            f"gives invalid coordinates (start_idx={start_idx}, "
            f"max={scores.shape[2]})."
        )
    scores = scores[:, :, start_idx:start_idx+zoom_n_bases, :]
    seqs_one_hot = seqs_one_hot[:, start_idx:start_idx+zoom_n_bases, :]

    seq_length = scores.shape[2]
    total_classes = scores.shape[1]
    total_sequences = seqs_one_hot.shape[0]
    total_plots = total_sequences * total_classes
    xlabel_list=[]

    # Set defaults
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

    # Handle sharey with create_plot now that we have a third option (which handles it elsewhere)
    kwargs['sharey'] = False if sharey == 'sequence' else sharey

    fig, axs = create_plot(
        ax=ax,
        kwargs_dict=kwargs,
        default_width=seq_length//10,
        default_height=0.5+2.25*total_plots,
        nrows=total_plots,
        default_sharex=False,
        default_sharey=False,
        default_h_pad=0.2,
    )

    if isinstance(axs, plt.Axes):
        axs = [axs]

    plot_idx = 0
    # Gradient/mutagenesis-letters logos are drawn via FastLogo, which benefits from batch
    # processing. FastLogo supports per-logo x-axis coordinates/strand, so every logo in
    # this call (across all sequences/classes) is deferred here and drawn as a single
    # batched FastLogo call below, regardless of differing coordinates/strands.
    pending_logos = []
    for seq_i in range(total_sequences):
        # Gather this sequence's seq and score values
        seq_x = seqs_one_hot[seq_i, ...]
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
        data_range = np.abs(np.nanmax(seq_scores) - np.nanmin(seq_scores))
        sequence_min = np.nanmin(seq_scores) - 0.25*data_range
        sequence_max = np.nanmax(seq_scores) + 0.25*data_range

        # Parse coordinates if supplied
        if coordinates is not None:
            chrom, start, _, strand = _parse_coordinates_input(coordinates[seq_i])
            if zoom_n_bases is not None:
                start += start_idx
                end = start + zoom_n_bases
            left, right = (end, start) if strand == "-" else (start, end)
            default_xlabel = f"{start:,.0f}-{end:,.0f}:{strand} ({np.abs(end - start)} bp)"
            for _ in range(total_classes-1):
                xlabel_list.append(None) # Add empty labels to all but non-final plot for sequence
            xlabel_list.append(default_xlabel)
        else:
            left, right = 0, seq_length

        reversed_positions = left > right
        positions_for_seq = np.arange(left, right, -1 if reversed_positions else 1)

        for class_i in range(total_classes):
            ax = axs[plot_idx]
            plot_idx += 1

            # Plot values for this sequence x class combo
            if method == "mutagenesis":
                _plot_mutagenesis_map(seq_scores[class_i], ax=ax, start=left, end=right, **plot_kws)
                # Handle layout
                if sharey == 'sequence':
                    ax.set_ylim([sequence_min, sequence_max])
                else:
                    ax.set_ymargin(0.25)
            else:
                # Defer drawing so every logo in this call gets batched into a single
                # FastLogo call below (positions/mirror_glyphs are per-logo, see
                # _build_attribution_logo); the ylim to apply afterwards must be
                # deferred with it (see note there).
                pending_logos.append({
                    "ax": ax,
                    "values": seq_scores[class_i],
                    "positions": positions_for_seq,
                    "reversed": reversed_positions,
                    "ylim": (sequence_min, sequence_max) if sharey == 'sequence' else None,
                })

            if class_labels is not None:
                # Plot at bottom half if mutagenesis scatter (usually negative values), top half for letters (usually positive)
                label_rel_y = 0.3 if method == 'mutagenesis' else 0.7
                ax.annotate(class_labels[class_i], (0.025, label_rel_y), xycoords= 'axes fraction', fontsize=16, ha="left", va="center")

            # Draw highlights
            if highlight_positions:
                for hl_start, hl_end in highlight_positions:
                    # Move highlights w.r.t zoom
                    if coordinates is None:
                        hl_start = hl_start - start_idx
                        hl_end = hl_end - start_idx
                    elif hl_end < (start_idx+zoom_n_bases): # Reverse compatibility: old idxes (0-indexed) with coordinates
                        # Handle reversed axis if negative strand, adding 1 to compensate for flipping start and end (which messes up -0.5 later)
                        if left > right:
                            hl_start =  left - (hl_start - start_idx) + 1
                            hl_end = left - (hl_end - start_idx) - 1
                        else:
                            hl_start = left + (hl_start - start_idx)
                            hl_end = left + (hl_end - start_idx)
                    # -0.5 on both to make it a half-open interval, aligning with indexing
                    ax.axvspan(
                        xmin=hl_start-0.5,
                        xmax=hl_end-0.5,
                        **highlight_kws
                    )

            # Set xtick behaviour
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_major_formatter("{x:,.0f}")

        # Set the title for the sequence (subplot)
        if sequence_labels:
            axs[plot_idx - total_classes].set_title(sequence_labels[seq_i], fontsize=14)

    # Draw all deferred gradient/mutagenesis-letters logos as a single batched FastLogo call,
    # regardless of differing coordinates/strands (FastLogo supports per-logo positions/mirror_glyphs)
    if pending_logos:
        positions_batch = np.stack([p["positions"] for p in pending_logos], axis=0)
        mirror_batch = np.array([p["reversed"] for p in pending_logos])
        values_batch = np.stack([p["values"] for p in pending_logos], axis=0)

        # `spines`/`figsize`/`rotate` are consumed here (drawing-time), the rest of
        # `plot_kws` is forwarded to `fast_logomaker.FastLogo` (batch-building-time).
        logo_kws = plot_kws.copy()
        spines_kw = logo_kws.pop('spines', True)
        figsize_kw = logo_kws.pop('figsize', (20, 1))
        rotate_kw = logo_kws.pop('rotate', False)

        logo = _build_attribution_logo(
            values_batch,
            alphabet=("A", "C", "G", "T"),
            positions=positions_batch,
            mirror_glyphs=mirror_batch,
            **logo_kws,
        )
        for idx, entry in enumerate(pending_logos):
            _draw_logo(
                logo,
                idx,
                ax=entry["ax"],
                return_ax=False,
                spines=spines_kw,
                figsize=figsize_kw,
                rotate=rotate_kw,
                reversed_positions=entry["reversed"],
            )
            entry["ax"].autoscale(enable=True, axis='y') # undo fixing of axes within FastLogo
            # Handle layout (deferred: must happen after the draw+autoscale above, not before)
            if entry["ylim"] is not None:
                entry["ax"].set_ylim(entry["ylim"])
            else:
                entry["ax"].set_ymargin(0.25)

    # Set xlabels with coordinates if not supplied
    if coordinates is not None and 'xlabel' not in kwargs:
        kwargs['xlabel'] = xlabel_list

    return render_plot(fig, axs, **kwargs)
