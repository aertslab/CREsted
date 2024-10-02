from importlib.util import find_spec
from loguru import logger

from ._contribution_scores import contribution_scores

def _optional_function_warning(*args, **kwargs):
    logger.error(
        "The requested functionality requires the 'tfmodisco' package, which is not installed. "
        "Please install it with `pip install crested[tfmodisco]`.",
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
            create_clustermap,
            modisco_results,
            plot_patterns,
            plot_similarity_heatmap,
            plot_pattern_instances ,
            plot_clustermap_tf_motif 
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise
else:
    create_clustermap = _optional_function_warning
    modisco_results = _optional_function_warning
    plot_patterns = _optional_function_warning
    plot_similarity_heatmap = _optional_function_warning
    plot_pattern_instances = _optional_function_warning 
    plot_clustermap_tf_motif = _optional_function_warning

# Export these functions for public use
__all__ = [
    "contribution_scores",
]

if MODISCOLITE_AVAILABLE:
    __all__.extend(
        [
            "create_clustermap",
            "modisco_results",
            "plot_patterns",
            "plot_similarity_heatmap",
            "plot_pattern_instances", 
            "plot_clustermap_tf_motif"
        ]
    )
