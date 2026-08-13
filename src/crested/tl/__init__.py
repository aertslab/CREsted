"""The tools module `crested.tl` provides everything you need to train and interpret models."""

from loguru import logger

# Setup backend before importing any keras-dependent modules
from crested._backend import _setup_backend

_setup_backend()

from . import data, losses, metrics, modisco, zoo  # noqa: E402
from ._configs import TaskConfig, default_configs  # noqa: E402
from ._crested import Crested  # noqa: E402
from ._old import enhancer_design_in_silico_evolution, enhancer_design_motif_insertion  # noqa: E402
from ._tools import (  # noqa: E402
    contribution_scores,
    contribution_scores_specific,
    evaluate,
    extract_layer_embeddings,
    predict,
    score_gene_locus,
)
