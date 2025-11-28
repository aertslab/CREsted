"""Import all submodules and setup logging."""

from importlib.metadata import version

# Import utils eagerly (needed for logging setup)
from . import utils

# Import lightweight modules
from ._datasets import get_dataset, get_model, get_motif_db
from ._genome import Genome, register_genome
from ._io import import_beds, import_bigwigs

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
