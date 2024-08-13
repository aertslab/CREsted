import os

# Set keras backend
try:
    import tensorflow as tf

    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
except ImportError:
    try:
        import torch

        os.environ["KERAS_BACKEND"] = "torch"
    except ImportError as e:
        raise ImportError(
            "No backend found. Please install either tensorflow or pytorch."
        ) from e

import sys
from importlib.metadata import version

from . import pl, pp, tl
from ._datasets import get_dataset, get_motif_db
from ._io import import_beds, import_bigwigs
from ._logging import setup_logging

__all__ = [
    "pl",
    "pp",
    "tl",
    "import_beds",
    "import_bigwigs",
    "setup_logging",
    "get_dataset",
    "get_motif_db",
]

__version__ = version("crested")

os.environ["AUTOGRAPH_VERBOSITY"] = "0"

# Setup loguru logging
setup_logging(log_level="INFO", log_file=None)
