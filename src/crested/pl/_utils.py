"""Utility functions for plotting in CREsted."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from crested.utils import parse_region
from crested.utils._logging import log_and_raise


def render_plot(
    fig: plt.Figure,
    axs: plt.Axes | list[plt.Axes],
    title: str | list[str] | None = None,
    suptitle: str | None = None,
    xlabel: str | list[str] | None = None,
    ylabel: str | list[str] | None = None,
    supxlabel: str | None = None,
    supylabel: str | None = None,
    xlim: tuple[float, float]| list[tuple(float, float)] | None = None,
    ylim: tuple[float, float]| list[tuple(float, float)] | None = None,
    grid: bool | Literal['x', 'y', 'both'] = False,
    tight_layout: bool = False,
    tight_rect: tuple | None = None,
    title_fontsize: int = 16,
    suptitle_fontsize: int = 18,
    xlabel_fontsize: int = 14,
    ylabel_fontsize: int = 14,
    supxlabel_fontsize: int = 16,
    supylabel_fontsize: int = 16,
    xtick_fontsize: int = 12,
    ytick_fontsize: int = 12,
    xtick_rotation: float = 0.,
    ytick_rotation: float = 0.,
    xtick_ha: Literal['left', 'center', 'right'] | None = None,
    ytick_va: Literal['baseline', 'bottom', 'center', 'center_baseline', 'top'] | None = None,
    xtick_rotationmode: Literal['default', 'anchor'] | None = None,
    ytick_rotationmode: Literal['default', 'anchor'] | None = None,
    show: bool = True,
    save_path: str | None = None,
    x_label_fontsize = 'deprecated',
    y_label_fontsize = 'deprecated',
    x_tick_fontsize = 'deprecated',
    y_tick_fontsize = 'deprecated',
    x_label_rotation = 'deprecated',
    y_label_rotation = 'deprecated',
    xlabel_rotation = 'deprecated',
    ylabel_rotation = 'deprecated',
) -> None | (plt.Figure, plt.Axis) | (plt.Figure, list[plt.Axis]):
    """
    Render a plot with customization options.

    Note
    ----
    This function should never be called directly. Rather, the other plotting functions call this function.

    Parameters
    ----------
    fig
        The figure object to render.
    axs
        The axis object or list of axis objects to render.
    title
        Axis-level title of the plot. If a list, matched to each axis in axs; if a string, applied to all axes.
    xlabel
        Label for the X-axis. If a list, matched to each axis in axs; if a string, applied to all axes.
    ylabel
        Label for the Y-axis.  If a list, matched to each axis in axs; if a string, applied to all axes.
    supxlabel
        Suplabel for the X-axis.
    supylabel
        Suplabel for the Y-axis.
    xlim
        X-axis limits. If a list of lists, matched to each axis in axs; if a single list, applied to all axes.
    ylim
        Y-axis limits. If a list of lists, matched to each axis in axs; if a single list, applied to all axes.
    grid
        Add a major tick grid. True/'both' for a full grid, 'x' or 'y' for a specific axis, or False to disable.
    tight_layout
        Whether to run `fig.tight_layout()` after setting all plot properties. Default is False; constrained layout through `plt.subplots(layout='constrained')` is preferred.
    tight_rect
        Normalized coordinates in which subplots will fit, for `fig.tight_layout(tight_rect=tight_rect)`. Only does something if `tight_layout` is True.
    title_fontsize
        Font size for the title.
    xlabel_fontsize
        Font size for the X-axis labels.
    ylabel_fontsize
        Font size for the Y-axis labels.
    xtick_fontsize
        Font size for the X-axis tick labels.
    ytick_fontsize
        Font size for the Y-axis tick labels.
    xtick_rotation
        Rotation of the X-axis labels in degrees.
    ytick_rotation
        Rotation of the Y-axis labels in degrees.
    xtick_ha
        Horizontal alignment of the X-axis tick labels. If None, inferred to be appropriate for xtick_rotation.
    ytick_va
        Vertical alignment of the Y-axis tick labels. If None, kept unchanged.
    xtick_rotationmode
        Rotation mode when rotating the X-axis tick labels. If None, inferred to be appropriate for xtick_rotation.
    ytick_rotationmode
        Rotation mode when rotating the Y-axis tick labels. If None, kept unchanged.
    show
        Whether to display the plot. Set this to False if you want to return the figure object to customize it further.
    save_path
        Optional path to save the figure. If None, the figure is displayed but not saved.
    x_label_fontsize, y_label_fontsize, x_tick_fontsize, y_tick_fontsize, x_label_rotation, y_label_rotation
        Renamed arguments. Please use their `xlabel_*` or `ylabel_*` versions instead. For `x/y_label_rotation`, renamed to `[x/y]tick_rotation`.
    """
    # Check deprecated arguments
    if x_label_fontsize != 'deprecated':
        xlabel_fontsize = x_label_fontsize
        logger.warning("Argument `x_label_fontsize` is renamed; please use xlabel_fontsize instead.")
    if y_label_fontsize != 'deprecated':
        ylabel_fontsize = y_label_fontsize
        logger.warning("Argument `y_label_fontsize` is renamed; please use ylabel_fontsize instead.")
    if x_tick_fontsize != 'deprecated':
        xtick_fontsize = x_tick_fontsize
        logger.warning("Argument `x_tick_fontsize` is renamed; please use xtick_fontsize instead.")
    if y_tick_fontsize != 'deprecated':
        ytick_fontsize = y_tick_fontsize
        logger.warning("Argument `y_tick_fontsize` is renamed; please use ytick_fontsize instead.")
    if x_label_rotation != 'deprecated':
        xtick_rotation = x_label_rotation
        logger.warning("Argument `x_label_rotation` is renamed; please use xtick_rotation instead.")
    if y_label_rotation != 'deprecated':
        ytick_rotation = y_label_rotation
        logger.warning("Argument `y_label_rotation` is renamed; please use ytick_rotation instead.")
    if xlabel_rotation != 'deprecated':
        xtick_rotation = xlabel_rotation
        logger.warning("Argument `xlabel_rotation` is renamed; please use xtick_rotation instead.")
    if ylabel_rotation != 'deprecated':
        ytick_rotation = ylabel_rotation
        logger.warning("Argument `ylabel_rotation` is renamed; please use ytick_rotation instead.")

    @log_and_raise(ValueError)
    def _check_input_lengths():
        if xlabel is not None and not isinstance(xlabel, str):
            if len(xlabel) != n_axes:
                raise ValueError(f"List of x labels provided, but number of x labels {len(xlabel)} does not match number of axes ({n_axes}).")
        if ylabel is not None and not isinstance(ylabel, str):
            if len(ylabel) != n_axes:
                raise ValueError(f"List of y labels provided, but number of y labels {len(ylabel)} does not match number of axes ({n_axes}).")
        if title is not None and not isinstance(title, str):
            if len(title) != n_axes:
                raise ValueError(f"List of axis titles provided, but number of titles {len(title)} does not match number of axes ({n_axes}).")
        if xlim is not None and isinstance(xlim[0], Sequence):
            if len(xlim) != n_axes:
                raise ValueError(f"List of xlims provided, but number of titles {len(xlim)} does not match number of axes ({n_axes}).")
        if ylim is not None and isinstance(ylim[0], Sequence):
            if len(ylim) != n_axes:
                raise ValueError(f"List of ylims provided, but number of titles {len(ylim)} does not match number of axes ({n_axes}).")

    # Handle axes
    if isinstance(axs, plt.Axes):
        axs = [axs]
    elif isinstance(axs, np.ndarray):
        axs = axs.ravel()
    n_axes = len(axs)

    # Check input
    _check_input_lengths()

    # Handle single-entry inputs
    if isinstance(xlabel, str):
        xlabel = [xlabel]*n_axes
    if isinstance(ylabel, str):
        ylabel = [ylabel]*n_axes
    if isinstance(title, str):
        title = [title]*n_axes
    if xlim is not None and not isinstance(xlim[0], Sequence):
        xlim = [xlim]*n_axes
    if ylim is not None and not isinstance(ylim[0], Sequence):
        ylim = [ylim]*n_axes

    # Infer downstream x rotation parameters
    if xtick_ha is None:
        if 0 < (xtick_rotation % 180) < 90:
            xtick_ha = 'right'
        elif 90 < (xtick_rotation % 180) < 180:
            xtick_ha = 'left'
        else:
            xtick_ha = 'center'
    if xtick_rotationmode is None:
        xtick_rotationmode =  'anchor' if (35 <= xtick_rotation <= 55) else 'default'

    # Handle grid alias
    if grid is True:
        grid = 'both'

    # Set figure labels
    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)
    if supxlabel:
        fig.supxlabel(supxlabel)
    if supylabel:
        fig.supylabel(supylabel)
    fig.supxlabel(fig.get_supxlabel(), fontsize=supxlabel_fontsize)
    fig.supylabel(fig.get_supylabel(), fontsize=supylabel_fontsize)
    fig.suptitle(fig.get_suptitle(), fontsize=suptitle_fontsize)

    # Set axis traits
    for i, ax in enumerate(axs):
        if xlabel is not None and xlabel[i] is not None:
            ax.set_xlabel(xlabel[i])
        if ylabel is not None and ylabel[i] is not None:
            ax.set_ylabel(ylabel[i])
        if title is not None and title[i] is not None:
            ax.set_title(title[i])
        ax.set_xlabel(ax.get_xlabel(), fontsize=xlabel_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ylabel_fontsize)
        ax.set_title(ax.get_title(), fontsize=title_fontsize)
        if xlim is not None and xlim[i] is not None:
            ax.set_xlim(xlim[i])
        if ylim is not None and ylim[i] is not None:
            ax.set_ylim(ylim[i])
        for label in ax.get_xticklabels():
            label.set_rotation(xtick_rotation)
            label.set_fontsize(xtick_fontsize)
            label.set_ha(xtick_ha)
            label.set_rotation_mode(xtick_rotationmode)
        for label in ax.get_yticklabels():
            label.set_fontsize(ytick_fontsize)
            label.set_rotation(ytick_rotation)
            if ytick_va is not None:
                label.set_va(ytick_va)
            if ytick_rotationmode is not None:
                label.set_rotation_mode(ytick_rotationmode)
        if grid:
            ax.grid(visible=True, axis=grid, color=".85")
            ax.set_axisbelow(True)

    # Set figure resizing
    if tight_layout:
        if tight_rect:
            fig.tight_layout(rect=tight_rect)
        else:
            fig.tight_layout()

    # Save and/or show and/or return
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    if not show and not save_path:
        return (fig, axs[0]) if len(axs) == 1 else (fig, axs)


def create_plot(
    ax: plt.Axes | None,
    kwargs_dict: dict,
    default_width: int = 8,
    default_height: int = 8,
    default_sharex: bool = False,
    default_sharey: bool = False,
    nrows: int = 1,
    ncols: int = 1,
    default_layout: Literal['constrained', 'compressed', 'tight', None] = 'constrained',
    default_h_pad: float | None = 0.2,
    default_w_pad: float | None = 0.2,
    **kwargs
) -> (plt.Figure, plt.Axes) | (plt.Figure, list[plt.Axes]):
    """
    Create a new plot or gather the figure if already existing.

    Effectively a wrapper around `plt.subplots` if there's no pre-existing axis, and a way to get the Figure object if there is.

    Note
    ----
    This function should never be called directly. Rather, the other plotting functions call this function.

    Parameters
    ----------
    ax
        A single axis object, or None. If an axis, will return its associated figure.
        If None, will create a new figure according to the other parameters.
    kwargs_dict
        The dictionary containing kwargs to set figure properties. Will consume arguments 'width', 'height', 'sharex', 'sharey'.
    default_width
        The default width of the figure, for `plt.subplots(figsize)`, if not in `kwargs_dict`.
    default_height
        The default height of the figure, for `plt.subplots(figsize)`, if not in `kwargs_dict`.
    default_sharex
        Default state of `plt.subplots(sharex)` if not in `kwargs_dict`.
    default_sharey
        Default state of `plt.subplots(sharey)` if not in `kwargs_dict`.
    nrows
        The number of rows for the subplot grid.
    ncols
        The number of columns for the subplot grid.
    default_layout
        The layout engine to use, if not in `kwargs_dict`. See :func:`~matplotlib.pyplot.figure` for more information.
    default_h_pad
        Minimum height padding to use between subplots, in inches, if not in `kwargs_dict`. See :meth:`~matplotlib.layout_engine.LayoutEngine.set()` for more information.
    default_w_pad
        Minimum width padding to use between subplots, in inches, if not in `kwargs_dict`. See :meth:`~matplotlib.layout_engine.LayoutEngine.set()` for more information.
    kwargs
        Extra kwargs passed to `plt.subplots`.
    """
    if ax is None:
        width = kwargs_dict.pop('width') if 'width' in kwargs_dict else default_width
        height = kwargs_dict.pop('height') if 'height' in kwargs_dict else default_height
        sharex = kwargs_dict.pop('sharex') if 'sharex' in kwargs_dict else default_sharex
        sharey = kwargs_dict.pop('sharey') if 'sharey' in kwargs_dict else default_sharey
        layout = kwargs_dict.pop('layout') if 'layout' in kwargs_dict else default_layout
        h_pad = kwargs_dict.pop('h_pad') if 'h_pad' in kwargs_dict else default_h_pad
        w_pad = kwargs_dict.pop('w_pad') if 'w_pad' in kwargs_dict else default_w_pad
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height), sharex=sharex, sharey=sharey, layout=layout, **kwargs)
        if layout is not None:
            fig.get_layout_engine().set(h_pad=h_pad, w_pad=w_pad)
    elif isinstance(ax, plt.Axes):
        for kwarg in ['width', 'height', 'sharex', 'sharey', 'layout', 'h_pad', 'w_pad']:
            if kwarg in kwargs_dict:
                logger.warning(f"Using keyword argument {kwarg} does not do anything when passing a pre-existing axis.")
        fig = ax.get_figure()
    else:
        raise ValueError(f"ax must be a single matplotlib ax or None, not {type(ax)}.")
    return fig, ax

def _parse_coordinates_input(
    coordinates: str | tuple[int, int] | tuple[str, int, int] | tuple[int, int, str] | tuple[str, int, int, str]
    ) -> tuple[str, int, int, str]:
    """Parse possible coordinates inputs.

    Parameters
    ----------
    coordinates
        A string or tuple of coordinates. Input possibilities:
        String: chr:start-end or chr:start-end:strand.
        Tuple: a (start, end), (chr, start, end), (start, end, string), (chrom, start, end, string).
        Distinguishes between the 3-len tuples by checking if last is a string, indicating a strand marker.

    Returns
    -------
    A (chrom, start, end, strand) tuple, with chrom set to None and strand set to + if not provided.
    """
    if isinstance(coordinates, str):
        chrom, start, end, strand = parse_region(coordinates)
    elif len(coordinates) == 2:
        start, end = coordinates
        chrom = None
        strand = "+"
    elif len(coordinates) == 3:
        if isinstance(coordinates[-1], str):
            start, end, strand = coordinates
            chrom = None
        else:
            chrom, start, end = coordinates
            strand = "+"
    elif len(coordinates) == 4:
        chrom, start, end, strand = coordinates
    return chrom, start, end, strand
