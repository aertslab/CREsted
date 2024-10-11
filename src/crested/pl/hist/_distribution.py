"""Distribution plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from loguru import logger

from crested.pl._utils import render_plot
from crested.utils._logging import log_and_raise


def distribution(
    adata: AnnData,
    target: str = "groundtruth",
    class_names: list[str] | None = None,
    split: str | None = None,
    log_transform: bool = True,
    share_y: bool = False,
    **kwargs,
) -> plt.Figure:
    """
    Histogram of region distribution for specified classes.

    Parameters
    ----------
    adata
        AnnData object containing the predictions in `layers`.
    target
        The target to plot the distribution for, either "groundtruth" or the name of a prediction layer in adata.layers.
    class_names
        List of classes in `adata.obs`. If None, will create a plot per class in `adata.obs`.
    split
        'train', 'val', 'test' subset or None. If None, will use all targets. If not None, expects a "split" column in adata.var.
    log_transform
        Whether to log-transform the data before plotting.
    share_y
        Whether to share the y-axis across all plots.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.

    See Also
    --------
    crested.pl.render_plot

    Example
    --------
    >>> crested.pl.hist.distribution(
    ...     adata, split="test", share_y=False, class_names=["Astro", "Vip"]
    ... )

    .. image:: ../../../../docs/_static/img/examples/hist_distribution.png
    """

    @log_and_raise(ValueError)
    def _check_input_params():
        if class_names is not None:
            for class_name in class_names:
                if class_name not in list(adata.obs_names):
                    raise ValueError(f"{class_name} not found in adata.obs_names.")

        if target not in ["groundtruth"] + list(adata.layers.keys()):
            raise ValueError(f"{target} not found in adata.layers.")

        if split is not None:
            if "split" not in adata.var:
                raise ValueError(
                    "No split column found in adata.var. Run `pp.train_val_test_split` first if 'split' is not None."
                )

    _check_input_params()

    if class_names is None:
        class_names = list(adata.obs_names)

    logger.info(f"Plotting histograms for target: {target}, classes: {class_names}")

    n_classes = len(class_names)
    n_cols = int(np.ceil(np.sqrt(n_classes)))
    n_rows = int(np.ceil(n_classes / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(kwargs.get("width", 8) * n_cols, kwargs.get("height", 6) * n_rows),
        sharex=True,
        sharey=share_y,
    )
    axes = axes.flatten() if n_classes > 1 else [axes]

    for i, class_name in enumerate(class_names):
        ax = axes[i]

        if target == "groundtruth":
            data = adata.X[adata.obs_names.get_loc(class_name), :]
        else:
            data = adata.layers[target][adata.obs_names.get_loc(class_name), :]

        if log_transform:
            data = np.log(data + 1)

        if split is not None:
            data = data[adata.var["split"] == split]

        sns.histplot(
            data,
            kde=True,
            ax=ax,
            color="skyblue",
            binwidth=np.ptp(data) / 50,
            stat="frequency",
        )
        ax.set_title(class_name)
        ax.grid(True)

    default_height = 6 * n_rows
    default_width = 8 * n_cols

    if "width" not in kwargs:
        kwargs["width"] = default_width
    if "height" not in kwargs:
        kwargs["height"] = default_height

    return render_plot(fig, **kwargs)
