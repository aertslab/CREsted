"""General utility functions used across CREsted."""

from ._logging import setup_logging
from ._old import (
    EnhancerOptimizer,
    derive_intermediate_sequences,
)
from ._seq_utils import (
    flip_region_strand,
    hot_encoding_to_sequence,
    one_hot_encode_sequence,
    parse_region,
    reverse_complement,
)
from ._utils import (
    calculate_nucleotide_distribution,
    fetch_sequences,
    read_bigwig_region,
)

# Lazy imports for functions that require keras
__all__ = [
    "setup_logging",
    "flip_region_strand",
    "hot_encoding_to_sequence",
    "one_hot_encode_sequence",
    "parse_region",
    "reverse_complement",
    "calculate_nucleotide_distribution",
    "fetch_sequences",
    "read_bigwig_region",
    "permute_model",  # Lazy-loaded
    "load_model",  # Lazy-loaded
]


# Lazy import keras-dependent functions
_LAZY_FUNCTIONS = {
    "permute_model": '._model_utils',
    "load_model": '._model_utils',
}

def __getattr__(name):
    """Lazy import certain functions only when accessed."""
    if name in _LAZY_FUNCTIONS:
        import importlib
        module = importlib.import_module(_LAZY_FUNCTIONS[name], __name__)
        func = getattr(module, name)
        globals()[name] = func
        return func
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

