"""Tn5 bias prediction plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from crested.pl._utils import render_plot


def tn5_bias(
    predicted_biases: np.ndarray,
    share_y: bool = True,
    **kwargs,
) -> plt.Figure:
    """
    Plot predicted Tn5 biases from :func:`~crested.tl.tn5_bias_prediction` as subplots.

    Each subplot represents a region and the x-axis represents the position in the region.

    Parameters
    ----------
    predicted_biases
        2D array of predicted biases, where each row represents a region.
    share_y
        If True, share the y-axis across all subplots.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.

    Returns
    -------
    plt.Figure
        The generated figure with subplots of predicted biases.

    See Also
    --------
    crested.pl.render_plot
    crested.tl.tn5_bias_prediction

    Examples
    --------
    >>> tn5_bias_model = crested.get_model("tn5_bias")
    >>> regions = ["chr1:1000000-1000200", "chr1:1000800-1001000"]
    >>> region_seqs = crested.utils.fetch_sequences(regions, genome_path)
    >>> predicted_biases = crested.tl.tn5_bias_prediction(region_seqs, tn5_bias_model)
    >>> crested.pl.hist.predicted_biases(
    ...     predicted_biases, title="Predicted Tn5 biases", height=6
    ... )

    .. image:: ../../../../docs/_static/img/examples/hist_tn5_bias.png
    """
    n_regions = predicted_biases.shape[0]

    fig, axes = plt.subplots(
        n_regions,
        1,
        sharex=True,
        sharey=share_y,
        dpi=200,
    )

    axes = axes.flatten() if n_regions > 1 else [axes]

    for i, region_biases in enumerate(predicted_biases):
        ax = axes[i]
        ax.bar(np.arange(len(region_biases)), region_biases, color="skyblue")
        # ax.set_xlabel("")
        ax.grid(True)
    fig.supylabel("Predicted bias")
    fig.supxlabel("Position (bp)")

    # Hide any unused axes
    for i in range(n_regions, len(axes)):
        axes[i].axis("off")

    return render_plot(fig, **kwargs)
