from loguru import logger

from ._contribution_scores import contribution_scores
from ._enhancer_design import enhancer_design_steps_contribution_scores


def _optional_function_warning(*args, **kwargs):
    logger.error(
        "The requested functionality requires the 'tfmodisco' package, which is not installed. "
        "Please install it with `pip install crested[tfmodisco]`.",
    )


try:
    from ._modisco_results import (
        create_clustermap,
        modisco_results,
        plot_patterns,
        plot_similarity_heatmap,
    )
except ImportError:
    modisco_results = _optional_function_warning
    create_clustermap = _optional_function_warning
    plot_patterns = _optional_function_warning
    plot_similarity_heatmap = _optional_function_warning

if modisco_results is not None:
    __all__ = [
        "contribution_scores",
        "modisco_results",
        "create_clustermap",
        "plot_patterns",
        "plot_similarity_heatmap",
    ]
else:
    __all__ = ["contribution_scores"]
