

from loguru import logger

from crested.tl.design import in_silico_evolution, motif_insertion


def enhancer_design_in_silico_evolution(*args, **kwargs):
    """
    Create synthetic enhancers for a specified class using in silico evolution (ISE).

    Deprecated in favor of :func:`~crested.tl.design.in_silico_evolution`.

    :meta private:
    """
    logger.info(
        "`crested.tl.enhancer_design_in_silico_evolution` has been renamed to `crested.pl.design.in_silico_evolution` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return in_silico_evolution(*args, **kwargs)

def enhancer_design_motif_insertion(*args, **kwargs):
    """
    Create synthetic enhancers using motif insertions.

    Deprecated in favor of :func:`~crested.tl.design.motif_insertion`.

    :meta private:
    """
    logger.info(
        "`crested.tl.enhancer_design_motif_insertion` has been renamed to `crested.pl.design.motif_insertion` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return motif_insertion(*args, **kwargs)
