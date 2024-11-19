from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def coverage_predictions(
        adata
):
    """
    Barplots of all predictions in .layers vs the groundtruth for a specific region across comparing classes.

    Parameters
    ----------
    adata
        AnnData object containing the predictions in `layers`.
    region
        String in the format 'chr:start-end' representing the genomic location.
    model_names
        List of model names in `adata.layers`. If None, will create a plot per model in `adata.layers`.
    share_y
        Whether to rescale the y-axis to be the same across plots. Default is True.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to
        control the final plot output. Please see :func:`~crested.pl.render_plot`
        for details.

    See Also
    --------
    crested.pl.render_plot

    Example
    -------
    TODO: [wip]
    """


def coverage():
    """"TODO: update"""
