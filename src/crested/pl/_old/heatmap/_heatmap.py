
from loguru import logger

import crested


def correlations_predictions(*args, **kwargs):
    """
    Heatmap of correlations and ground truth, comparing all classes with each other.

    Deprecated in favor of :func:`~crested.pl.corr.heatmap`.

    :meta private:
    """
    logger.info(
        "`crested.pl.heatmap.correlations_predictions` has been renamed to `crested.pl.corr.heatmap` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.corr.heatmap(*args, **kwargs)


def correlations_self(*args, **kwargs):
    """
    Heatmap of self-correlation between the ground truth classes, comparing all classes with each other.

    Deprecated in favor of :func:`~crested.pl.corr.heatmap_self`.

    :meta private:
    """
    logger.info(
        "`crested.pl.heatmap.correlations_self` has been renamed to `crested.pl.corr.heatmap_self` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.corr.heatmap_self(*args, **kwargs)
