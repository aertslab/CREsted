import os
import sys
from importlib.metadata import version

from . import pl, pp, tl
from ._io import import_bigwigs, import_topics
from ._logging import setup_logging

__all__ = ["pl", "pp", "tl", "import_topics", "import_bigwigs", "setup_logging"]

__version__ = version("crested")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

# Setup loguru logging
setup_logging(log_level="INFO", log_file=None)
