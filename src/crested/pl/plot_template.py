from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from crested.pl._utils import create_plot, render_plot
from crested.utils._logging import log_and_raise


def prediction(
    prediction: np.ndarray,
    classes: list[str],
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Bar plot for a single prediction, comparing different classes.

    Parameters
    ----------
    prediction
        An array containing the prediction values for each class or cell type. It is squeezed to remove 1-wide dimensions if necessary.
    classes
        A list of class or cell type labels corresponding to the predictions.
    plot_kws
        Extra keyword arguments passed to :func:`~matplotlib.Axes.bar`. Defaults: `'alpha': 0.8`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width, height
        Dimensions of the newly created figure if `ax=None`. Default is (18, 3).
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Custom defaults for `prediction`: `xlabel='Cell types'`, `ylabel='Prediction'`, `grid="y"`.

    See Also
    --------
    crested.pl.render_plot
    crested.pl.bar.region
    crested.pl.bar.region_predictions

    Example
    -------
    >>> crested.pl.bar.prediction(
    ...     pred,
    ...     classes=list(adata.obs_names),
    ...     title="Region chr1:3094805-3095305"
    ... )
    """
    # Check inputs
    @log_and_raise(ValueError)
    def _check_input_params():
        if len(prediction) != len(classes):
            raise ValueError(
                f"The length of prediction array ({len(prediction)}) must match the number of classes ({len(classes)})."
            )

    # Ensure the prediction array is 1-dimensional
    prediction = prediction.squeeze()

    _check_input_params()

    # Set defaults
    plot_width = kwargs.pop('width') if 'width' in kwargs else 18
    plot_height = kwargs.pop('height') if 'height' in kwargs else 3
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = "Cell types"
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = "Prediction"
    if 'grid' not in kwargs:
        kwargs['grid'] = 'y'
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'alpha' not in plot_kws:
        plot_kws['alpha'] = 0.8

    # Create plot
    fig, ax = create_plot(ax=ax, width=plot_width, height=plot_height)
    ax.bar(classes, prediction, **plot_kws)

    # Use render_plot to finalize and return the figure
    return render_plot(fig, ax, **kwargs)
