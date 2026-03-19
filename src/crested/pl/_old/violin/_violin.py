from loguru import logger

import crested


def correlations(*args, **kwargs):
    """
    Plot the distribution of class correlations per model.

    Deprecated in favor of :func:`~crested.pl.corr.violin`.

    :meta private:
    """
    logger.info(
        "`crested.pl.violin.correlations` has been renamed to `crested.pl.corr.violin` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.corr.violin(*args, **kwargs)
