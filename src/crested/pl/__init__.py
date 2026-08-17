"""
The plotting module `crested.pl` provides a variety of functions to visualize your results.

It's organised into submodules, according to what kind of data you would like to plot.
"""

from loguru import logger

from . import corr, design, dist, explain, locus, modisco, qc, region
from ._old import bar, heatmap, hist, patterns, scatter, violin
from ._utils import create_plot, render_plot
