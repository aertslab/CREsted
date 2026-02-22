from loguru import logger

import crested


def enhancer_design_steps_contribution_scores(*args, **kwargs):
    """
    Plot contribution scores, marking the changed nucleotides at each step.

    Deprecated in favor of :func:`~crested.pl.design.step_contribution_scores`.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.enhancer_design_steps_contribution_scores` has been renamed to `crested.pl.design.step_contribution_scores` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.design.step_contribution_scores(*args, **kwargs)

def enhancer_design_steps_predictions(*args, **kwargs):
    """
    Plot the development of scores over each step of enhancer design.

    Deprecated in favor of :func:`~crested.pl.design.step_predictions`.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.enhancer_design_steps_predictions` has been renamed to `crested.pl.design.step_predictions` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.design.step_predictions(*args, **kwargs)
