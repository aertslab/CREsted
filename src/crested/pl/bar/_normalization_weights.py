"""Bar plot of normalization weights."""

from __future__ import annotations

import matplotlib.pyplot as plt
from anndata import AnnData

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


def normalization_weights(
    adata: AnnData,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Plot the distribution of normalization scaling factors per cell type.

    Parameters
    ----------
    adata
        AnnData object containing the normalization weights in `obsm["weights"]`.
    plot_kws
        Extra keyword arguments passed to :meth:`~matplotlib.axes.Axes.bar`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width, height
        Dimensions of the newly created figure if `ax=None`. Default is (11, 8).
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `normalization_weights`: `xlabel="Cell type"`, `ylabel="Scaling factor"`, `grid='y'`.

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
    # Check inputs
    @log_and_raise(ValueError)
    def _check_input_params():
        if "weights" not in adata.obsm:
            raise ValueError("Normalization weights not found in adata.obsm['weights']")
    _check_input_params()

    # Set defaults
    plot_width = kwargs.pop('width') if 'width' in kwargs else 20
    plot_height = kwargs.pop('height') if 'height' in kwargs else 3
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = "Cell type"
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = "Scaling factor"
    if 'grid' not in kwargs:
        kwargs['grid'] = 'y'
    plot_kws = {} if plot_kws is None else plot_kws.copy()

    # Gather data
    classes = list(adata.obs_names)
    weights = adata.obsm["weights"]

    # Plot
    fig, ax = create_plot(ax=ax, width=plot_width, height=plot_height)
    ax.bar(classes, weights, **plot_kws)
    return render_plot(fig, ax, **kwargs)
