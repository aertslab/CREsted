"""Init file for the patterns module."""

from importlib.util import find_spec

from loguru import logger

from ._contribution_scores import contribution_scores
from ._enhancer_design import (
    enhancer_design_steps_contribution_scores,
    enhancer_design_steps_predictions,
)


def _optional_function_warning(*args, **kwargs):
    logger.error(
        "The requested functionality requires the 'tfmodisco' package, which is not installed. "
        "Please install it with `pip install modisco-lite>=2.2.1.",
    )


if find_spec("modiscolite") is not None:
    MODISCOLITE_AVAILABLE = True
else:
    MODISCOLITE_AVAILABLE = False

if MODISCOLITE_AVAILABLE:
    try:
        import modiscolite

        # Import all necessary functions from _modisco_results
        from ._modisco_results import (
            class_instances,
            clustermap,
            clustermap_tf_motif,
            clustermap_tomtom_similarities,
            clustermap_with_pwm_logos,
            modisco_results,
            selected_instances,
            similarity_heatmap,
            tf_expression_per_cell_type,
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise
else:
    clustermap = _optional_function_warning
    clustermap_tomtom_similarities = _optional_function_warning
    modisco_results = _optional_function_warning
    selected_instances = _optional_function_warning
    similarity_heatmap = _optional_function_warning
    class_instances = _optional_function_warning
    clustermap_tf_motif = _optional_function_warning
    tf_expression_per_cell_type = _optional_function_warning
    clustermap_with_pwm_logos = _optional_function_warning

# Export these functions for public use
__all__ = [
    "contribution_scores",
    "enhancer_design_steps_contribution_scores",
    "enhancer_design_steps_predictions",
]

if MODISCOLITE_AVAILABLE:
    __all__.extend(
        [
            "clustermap",
            "modisco_results",
            "class_instances",
            "similarity_heatmap",
            "selected_instances",
            "clustermap_tf_motif",
            "tf_expression_per_cell_type",
            "clustermap_with_pwm_logos",
            "clustermap_tomtom_similarities",
        ]
    )
