"""Import all modules and classes from the 'crested.tl' package."""

from importlib.util import find_spec

from loguru import logger

from . import data, losses, metrics, zoo
from ._configs import TaskConfig, default_configs
from ._crested import Crested
from ._tools import (
    contribution_scores,
    contribution_scores_specific,
    enhancer_design_in_silico_evolution,
    enhancer_design_motif_insertion,
    extract_layer_embeddings,
    predict,
    score_gene_locus,
)

if find_spec("modiscolite") is not None:
    MODISCOLITE_AVAILABLE = True
else:
    MODISCOLITE_AVAILABLE = False

if MODISCOLITE_AVAILABLE:
    try:
        import modiscolite

        from . import modisco
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise
else:
    logger.warning(
        "modiscolite is not installed, 'crested.tl.modisco' module will not be available."
    )


__all__ = [
    "data",
    "losses",
    "metrics",
    "zoo",
    "TaskConfig",
    "default_configs",
    "Crested",
    "extract_layer_embeddings",
    "predict",
    "contribution_scores",
    "contribution_scores_specific",
    "enhancer_design_in_silico_evolution",
    "enhancer_design_motif_insertion",
    "score_gene_locus",
]

if MODISCOLITE_AVAILABLE:
    __all__.extend("modisco")
