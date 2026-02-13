"""The tools module `crested.tl` provides everything you need to train and interpret models."""

import importlib.util

from loguru import logger

# Setup backend before importing any keras-dependent modules
from crested._backend import _setup_backend

_setup_backend()

from . import data, losses, metrics, zoo  # noqa: E402
from ._configs import TaskConfig, default_configs  # noqa: E402
from ._crested import Crested  # noqa: E402
from ._tools import (  # noqa: E402
    contribution_scores,
    contribution_scores_specific,
    enhancer_design_in_silico_evolution,
    enhancer_design_motif_insertion,
    extract_layer_embeddings,
    predict,
    score_gene_locus,
    test,
)

if importlib.util.find_spec("modiscolite") is not None:
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
        "modiscolite is not installed, 'crested.tl.modisco' module will not be available. "
        "Install with: pip install crested[motif]"
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
