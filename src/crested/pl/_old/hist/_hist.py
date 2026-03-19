from loguru import logger

import crested


def distribution(*args, **kwargs):
    """
    Plot the distribution of ground truth and/or model values.

    Deprecated in favor of :func:`~crested.pl.dist.histogram`.

    :meta private:
    """
    logger.info(
        "`crested.pl.hist.distribution` has been renamed to `crested.pl.dist.histogram` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.dist.histogram(*args, **kwargs)

def locus_scoring(*args, **kwargs):
    """
    Score a locus with the model stepwise and optionally compare to bigwig values.

    Deprecated in favor of :func:`~crested.pl.locus.locus_scoring`.

    :meta private:
    """
    logger.info(
        "`crested.pl.hist.locus_scoring` has been renamed to `crested.pl.locus.locus_scoring` in version 1.6.1"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.locus.locus_scoring(*args, **kwargs)
