from importlib.metadata import version

from . import pl, pp, tl
from ._io import import_topics

__all__ = ["pl", "pp", "tl", "import_topics"]

__version__ = version("EnhancerAI")
