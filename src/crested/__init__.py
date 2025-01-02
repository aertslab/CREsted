"""Import all submodules, set the backend, and setup logging."""

import os
import warnings

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

from . import pl, pp, tl, utils
from ._datasets import get_dataset, get_model, get_motif_db
from ._genome import Genome, register_genome
from ._io import import_beds, import_bigwigs

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

__version__ = version("crested")

os.environ["AUTOGRAPH_VERBOSITY"] = "0"

# Setup loguru logging
utils.setup_logging(log_level="INFO", log_file=None)
