from loguru import logger

import crested


def class_density(*args, **kwargs):
    """
    Plot the ground truth vs model predictions as a scatterplot.

    Deprecated in favor of :func:`~crested.pl.corr.scatter`.

    :meta private:
    """
    logger.info(
        "`crested.pl.scatter.class_density` has been renamed to `crested.pl.corr.scatter` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.corr.scatter(*args, **kwargs)