from loguru import logger


def EnhancerOptimizer(*args, **kwargs):
    """
    (Alias function for the) class to optimize the mutated sequence based on the original prediction.

    Deprecated in favor of :func:`~crested.tl.design.EnhancerOptimizer`.

    :meta private:
    """
    from crested.tl.design import EnhancerOptimizer as EnhancerOptimizerNew
    logger.info(
        "`crested.utils.EnhancerOptimizer` has been moved to `crested.pl.design.EnhancerOptimizer` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return EnhancerOptimizerNew(*args, **kwargs)

def _weighted_difference(*args, **kwargs):
    """
    Deprecated in favor of :func:`~crested.tl.design._optimizer._weighted_difference`.

    :meta private:
    """
    from crested.tl.design._utils import _weighted_difference as _weighted_difference_new
    logger.info(
        "`crested.utils._weighted_difference` has been moved to `crested.pl.design._optimizer._weighted_difference` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return _weighted_difference_new(*args, **kwargs)

def derive_intermediate_sequences(*args, **kwargs):
    """
    Derive intermediate sequences of enhancer design

    Moved to :func:`~crested.tl.design.derive_intermediate_sequences`.

    :meta private:
    """
    from crested.tl.design import derive_intermediate_sequences as derive_intermediate_sequences_new
    logger.info(
        "`crested.utils.derive_intermediate_sequences` has been renamed to `crested.pl.design.derive_intermediate_sequences` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return derive_intermediate_sequences_new(*args, **kwargs)



