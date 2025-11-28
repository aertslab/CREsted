"""Import all utility functions and classes."""

from ._logging import setup_logging
from ._seq_utils import (
    hot_encoding_to_sequence,
    one_hot_encode_sequence,
    reverse_complement,
)
from ._utils import (
    EnhancerOptimizer,
    calculate_nucleotide_distribution,
    derive_intermediate_sequences,
    extract_bigwig_values_per_bp,
    fetch_sequences,
    read_bigwig_region,
)

# Lazy imports for functions that require keras
__all__ = [
    "setup_logging",
    "hot_encoding_to_sequence",
    "one_hot_encode_sequence",
    "reverse_complement",
    "EnhancerOptimizer",
    "calculate_nucleotide_distribution",
    "derive_intermediate_sequences",
    "extract_bigwig_values_per_bp",
    "fetch_sequences",
    "read_bigwig_region",
    "permute_model",  # Lazy-loaded
]


def __getattr__(name):
    """Lazy import for keras-dependent utilities."""
    if name == "permute_model":
        from ._model_utils import permute_model

        globals()["permute_model"] = permute_model
        return permute_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
