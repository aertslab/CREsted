import os
import sys
from importlib.metadata import version

from . import pl, pp, tl
from ._io import import_topics
from ._logging import setup_logging

__all__ = ["pl", "pp", "tl", "import_topics", "setup_logging"]

__version__ = version("crested")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"
