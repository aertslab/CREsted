"""
The plotting module `crested.pl` provides a variety of functions to visualize your results.

It's organised into submodules, according to what kind of data you would like to plot.
"""

import importlib.util

from loguru import logger

from . import corr, design, dist, explain, locus, modisco, qc, region
from ._old import bar, heatmap, hist, patterns, scatter, violin
from ._utils import create_plot, render_plot

if importlib.util.find_spec("modiscolite") is not None:
    MODISCOLITE_AVAILABLE = True
else:
    MODISCOLITE_AVAILABLE = False

if MODISCOLITE_AVAILABLE:
    try:
        import modiscolite

        from . import modisco
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise
else:
    logger.warning(
        "modiscolite is not installed, 'crested.pl.modisco' module will not be available. "
        "Install with: pip install crested[motif]"
    )

__all__ = [
    # Submodules
    "corr",
    "design",
    "dist",
    "explain",
    "locus",
    "modisco",
    "qc",
    "region",
    # Individual functions
    "create_plot",
    "render_plot",
    # Old alias modules
    "bar",
    "heatmap",
    "hist",
    "patterns",
    "scatter",
    "violin"
]

if MODISCOLITE_AVAILABLE:
    __all__.extend("modisco")
