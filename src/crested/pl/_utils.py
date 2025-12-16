"""Utility functions for plotting in CREsted."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

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
    grid: Literal[False, 'x', 'y', 'both'] = False,
    tight_rect: tuple | None = None,
    title_fontsize: int = 16,
    suptitle_fontsize: int = 18,
    xlabel_fontsize: int = 14,
    ylabel_fontsize: int = 14,
    supxlabel_fontsize: int = 16,
    supylabel_fontsize: int = 16,
    xtick_fontsize: int = 12,
    ytick_fontsize: int = 12,
    xlabel_rotation: int = 0,
    ylabel_rotation: int = 0,
    xlabel_ha: str = None,
    xlabel_rotationmode: str = None,
    show: bool = True,
    save_path: str | None = None,
    x_label_fontsize = 'deprecated',
    y_label_fontsize = 'deprecated',
    x_tick_fontsize = 'deprecated',
    y_tick_fontsize = 'deprecated',
    x_label_rotation = 'deprecated',
    y_label_rotation = 'deprecated',
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
        Add a major tick grid. Can be 'x', 'y', or 'both' to determine axis, True as alias for 'all', or False to disable.
    tight_rect
        Normalized coordinates in which subplots will fit.
    title_fontsize
        Font size for the title.
    xlabel_fontsize
        Font size for the X-axis labels.
    ylabel_fontsize
        Font size for the Y-axis labels.
    xtick_fontsize
        Font size for the X-axis ticks.
    ytick_fontsize
        Font size for the Y-axis ticks
    xlabel_rotation
        Rotation of the X-axis labels in degrees.
    ylabel_rotation
        Rotation of the Y-axis labels in degrees.
    xlabel_ha
        Horizontal alignment of the X-axis labels. If None, inferred to be appropriate for xlabel_rotation.
    xlabel_rotationmode
        Rotation mode when rotating the X-axis labels. If None, inferred to be appropriate for xlabel_rotation.
    show
        Whether to display the plot. Set this to False if you want to return the figure object to customize it further.
    save_path
        Optional path to save the figure. If None, the figure is displayed but not saved.
    x_label_fontsize, y_label_fontsize, x_tick_fontsize, y_tick_fontsize, x_label_rotation, y_label_rotation
        Renamed arguments. Please use their `xlabel_*` or `ylabel_*` instead.
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
        xlabel_rotation = x_label_rotation
        logger.warning("Argument `x_label_rotation` is renamed; please use xlabel_rotation instead.")
    if y_label_rotation != 'deprecated':
        ylabel_rotation = y_label_rotation
        logger.warning("Argument `y_label_rotation` is renamed; please use ylabel_rotation instead.")

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
    if xlabel_ha is None:
        if 0 < (xlabel_rotation % 180) < 90:
            x_label_ha = 'right'
        elif 90 < (xlabel_rotation % 180) < 180:
            x_label_ha = 'left'
        else:
            x_label_ha = 'center'
    if xlabel_rotationmode is None:
        xlabel_rotationmode =  'anchor' if (35 <= xlabel_rotation <= 55) else 'default'

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
    fig.ylabel(fig.get_supylabel(), fontsize=supylabel_fontsize)
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
            label.set_rotation(xlabel_rotation)
            label.set_fontsize(xtick_fontsize)
            label.set_ha(x_label_ha)
            label.set_rotation_mode(xlabel_rotationmode)
        for label in ax.get_yticklabels():
            label.set_fontsize(ytick_fontsize)
            label.set_rotation(ylabel_rotation)
        if grid:
            ax.grid(visible=True, axis=grid, color=".85")
            ax.set_axisbelow(True)

    # Set figure resizing
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
    width: int = 8,
    height: int = 8,
    nrows: int = 1,
    ncols: int = 1,
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
    width
        The width of the figure, for `plt.subplots(figsize)`.
    height
        The height of the figure, for `plt.subplots(figsize)`.
    nrows
        The number of rows for the subplot grid.
    ncols
        The number of columns for the subplot grid.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height), **kwargs)
    elif isinstance(ax, plt.Axes):
        fig = ax.get_figure()
    else:
        raise ValueError(f"ax must be a single matplotlib ax or None, not {type(ax)}.")
    return fig, ax
