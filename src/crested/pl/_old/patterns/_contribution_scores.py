from loguru import logger

import crested


def contribution_scores(*args, **kwargs):
    """
    Plot contribution scores for a (set of) sequence(s).

    Deprecated in favor of :func:`~crested.pl.explain.contribution_scores`.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.contribution_scores` has been renamed to `crested.pl.explain.contribution_scores` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.explain.contribution_scores(*args, **kwargs)
