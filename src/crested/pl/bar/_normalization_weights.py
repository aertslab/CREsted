"""Bar plot of normalization weights."""

from __future__ import annotations

import matplotlib.pyplot as plt
from anndata import AnnData

from crested._logging import log_and_raise
from crested.pl._utils import render_plot


def normalization_weights(adata: AnnData, **kwargs) -> plt.Figure:
    """
    Plot the distribution of normalization scaling factors per cell type.

    Parameters
    ----------
    adata
        AnnData object containing the normalization weights in `obsm["weights"]`.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to
        control the final plot output. Please see :func:`~crested.pl.render_plot`
        for details.

    See Also
    --------
    crested.pl.render_plot

    Example
    -------
    >>> crested.pl.bar.normalization_weights(
    ...     adata,
    ...     xlabel="Cell type",
    ...     ylabel="Scaling factor",
    ...     width=20,
    ...     height=3,
    ...     title="Normalization scaling factors",
    ... )

    .. image:: ../../../../docs/_static/img/examples/bar_normalization_weights.png
    """

    @log_and_raise(ValueError)
    def _check_input_params():
        if "weights" not in adata.obsm:
            raise ValueError("Normalization weights not found in adata.obsm['weights']")

    _check_input_params()

    weights = adata.obsm["weights"]
    classes = list(adata.obs_names)

    fig, ax = plt.subplots()
    ax.bar(classes, weights)

    # default plot size
    default_width = 20
    default_height = 3

    if "width" not in kwargs:
        kwargs["width"] = default_width
    if "height" not in kwargs:
        kwargs["height"] = default_height

    return render_plot(fig, **kwargs)
