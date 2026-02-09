"""Redirections for the new plotting function"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from loguru import logger

import crested
from crested.pl.region import bar


def normalization_weights(*args, **kwargs):
    """
    Bar plot for normalization weights.

    Deprecated in favor of :func:`~crested.pl.qc.normalization_weights`.

    :meta private:
    """
    logger.info(
        "`crested.pl.bar.normalization_weights` has been renamed to `crested.pl.qc.normalization_weights` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.qc.normalization_weights(*args, **kwargs)

def region(
    adata: AnnData | np.ndarray,
    region: str | None = None,
    target: str = "truth",
    classes: list[str] | None = None,
    log_transform: bool = False,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs
):
    """
    Bar plot for a single region, comparing different classes.

    Deprecated in favor of :func:`~crested.pl.region.bar`. Please use `crested.pl.region.bar(adata, region, model_names=model_name)` or `crested.pl.region.bar(adata, region, model_names='truth')` instead.

    :meta private:
    """
    # Deprecation warnings
    target_string = f"'{target}'"
    logger.warning(
        "region is deprecated since version 2.0.0 as its functionality is moved into `bar`. "
        f"Please use `crested.pl.region.bar(adata, region, model_names={target_string}, **kwargs)` instead."
    )

    return bar( # TODO: adjust when renamed
        data=adata,
        region=region,
        model_names=target,
        classes=classes,
        log_transform=log_transform,
        plot_kws=plot_kws,
        pred_color='tab:blue',
        truth_color='tab:blue',
        ax=ax,
        **kwargs
    )

def region_predictions(
    adata: AnnData,
    region: str,
    model_names: str | list[str] | None = None,
    share_y: bool = True,
    plot_kws: dict | None = None,
    **kwargs
):
    """
    Barplots of all predictions in .layers vs the ground truth for a specific region across comparing classes.

    Deprecated in favor of :func:`~crested.pl.region.bar`. Please use `crested.pl.region.bar(adata, region, model_names=None)` or `crested.pl.region.bar(adata, region, model_names=['model_name', 'truth'])` instead.

    :meta private:
    """
    # Parse model_names into targets tuple
    if model_names is not None:
        if isinstance(model_names, str):
            model_names = [model_names]
        model_names = [*model_names, 'truth']
    else:
        model_names = [*adata.layers.keys(), 'truth']

    # Deprecation warnings
    targets_string = "None" if model_names is None else model_names
    logger.warning(
        "region_predictions is deprecated since version 2.0.0 as its functionality is moved into `bar`. "
        f"Please use `crested.pl.region.bar(adata, region, model_names={targets_string}, **kwargs)` instead."
    )
    if 'share_y' in kwargs:
        kwargs['sharey'] = kwargs.pop('share_y')
        logger.warning("Argument `share_y` is deprecated since version 2.0.0; please use sharey instead to align with matplotlib.")

    # Mimic old behavior of names at ylabel rather than at title
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = ['Ground truth' if target == 'truth' else target for target in model_names]
    if 'title' not in kwargs:
        kwargs['title'] = None

    return bar(
        data=adata,
        region=region,
        model_names=model_names,
        plot_kws=plot_kws,
        sharey=share_y,
        **kwargs
    )

def prediction(
    prediction: np.ndarray,
    classes: list[str],
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Bar plot for a single prediction, comparing different classes.

    Deprecated in favor of :func:`~crested.pl.region.bar`. Please use `crested.pl.region.bar(data=prediction, classes=classes)` instead.

    :meta private:
    """
    logger.warning(
        "`prediction` is deprecated since version 2.0.0 as its functionality is moved into `crested.pl.region.bar`. "
        "Please use `crested.pl.region.bar(data=prediction, classes=classes, **kwargs)` instead."
    )
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'xlabel' in kwargs:
        logger.warning(f"'xlabel' is deprecated since v2.0.0. Please set it in 'plot_kws' instead: plot_kws={{xlabel={kwargs['xlabel']}}}")
        plot_kws['xlabel'] = kwargs.pop('xlabel')
    if 'ylabel' in kwargs:
        logger.warning(f"'ylabel' is deprecated since v2.0.0. Please set it in 'plot_kws' instead: plot_kws={{ylabel={kwargs['ylabel']}}}")
        plot_kws['ylabel'] = kwargs.pop('ylabel')
    if 'title' in kwargs:
        logger.warning(f"'title' is deprecated since v2.0.0. Please set it in 'plot_kws' instead: plot_kws={{title={kwargs['title']}}} or {{suptitle={kwargs['title']}}}")
        plot_kws['title'] = kwargs.pop('title')

    return bar(
        data=prediction,
        classes=classes,
        plot_kws=plot_kws,
        ax=ax,
        **kwargs
    )
