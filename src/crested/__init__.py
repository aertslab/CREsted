"""Import all submodules, set the backend, and setup logging."""

import os
import sys
import warnings
from importlib.metadata import version
from importlib.util import find_spec


def _setup_backend():
    """Set up Keras backend. Only called when actually needed."""
    if "KERAS_BACKEND" in os.environ:
        return  # Already set by user or previous call

    # Check which backend is available without importing
    if find_spec("tensorflow") is not None:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["AUTOGRAPH_VERBOSITY"] = "0"
    elif find_spec("torch") is not None:
        os.environ["KERAS_BACKEND"] = "torch"
    else:
        raise ImportError(
            "No backend found. Please install either tensorflow or pytorch."
        )


# Set backend early, but don't import TensorFlow/PyTorch yet
_setup_backend()

# Import utils eagerly (needed for logging setup)
from . import utils  # noqa: E402

# Import lightweight modules
from ._datasets import get_dataset, get_model, get_motif_db  # noqa: E402
from ._genome import Genome, register_genome  # noqa: E402
from ._io import import_beds, import_bigwigs  # noqa: E402

__version__ = version("crested")

# Setup loguru logging
utils.setup_logging(log_level="INFO", log_file=None)

# Lazy import heavy modules (pl, pp, tl)
_LAZY_MODULES = {
    "pl": ".pl",
    "pp": ".pp",
    "tl": ".tl",
}


def __getattr__(name):
    """Lazy import submodules only when accessed."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Add lazy modules to dir() output."""
    return sorted(list(globals().keys()) + list(_LAZY_MODULES.keys()))


__all__ = [
    "pl",
    "pp",
    "tl",
    "utils",
    "import_beds",
    "import_bigwigs",
    "get_dataset",
    "get_motif_db",
    "get_model",
    "Genome",
    "register_genome",
]
