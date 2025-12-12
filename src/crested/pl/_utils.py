"""Utility functions for plotting in CREsted."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def render_plot(
    fig: plt.Figure,
    axs: plt.Axes | list[plt.Axes],
    title: str | None = None,
    suptitle: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    supxlabel: str | None = None,
    supylabel: str | None = None,
    tight_rect: tuple | None = None,
    title_fontsize: int = 16,
    suptitle_fontsize: int = 20,
    x_label_fontsize: int = 14,
    y_label_fontsize: int = 14,
    x_tick_fontsize: int = 12,
    y_tick_fontsize: int = 12,
    x_label_rotation: int = 0,
    y_label_rotation: int = 0,
    x_label_ha: str = None,
    x_label_rotationmode: str = None,
    show: bool = True,
    save_path: str | None = None,
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
        Title of the plot.
    xlabel
        Label for the X-axis.
    ylabel
        Label for the Y-axis.
    supxlabel
        Suplabel for the X-axis.
    supylabel
        Suplabel for the Y-axis.
    tight_rect
        Normalized coordinates in which subplots will fit.
    title_fontsize
        Font size for the title.
    x_label_fontsize
        Font size for the X-axis labels.
    y_label_fontsize
        Font size for the Y-axis labels.
    x_tick_fontsize
        Font size for the X-axis ticks.
    y_tick_fontsize
        Font size for the Y-axis ticks
    x_label_rotation
        Rotation of the X-axis labels in degrees.
    y_label_rotation
        Rotation of the Y-axis labels in degrees.
    x_label_ha
        Horizontal alignment of the X-axis labels. If None, inferred to be appropriate for x_label_rotation.
    x_label_rotationmode
        Rotation mode when rotating the X-axis labels. If None, inferred to be appropriate for x_label_rotation.
    show
        Whether to display the plot. Set this to False if you want to return the figure object to customize it further.
    save_path
        Optional path to save the figure. If None, the figure is displayed but not saved.
    """
    if x_label_ha is None:
        if 0 < (x_label_rotation % 180) < 90:
            x_label_ha = 'right'
        elif 90 < (x_label_rotation % 180) < 180:
            x_label_ha = 'left'
        else:
            x_label_ha = 'center'
    if x_label_rotationmode is None:
        x_label_rotationmode =  'anchor' if (35 <= x_label_rotation <= 55) else 'default'

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)
    if supxlabel:
        fig.supxlabel(supxlabel)
    if supylabel:
        fig.supylabel(supylabel)

    # TODO: handle lists of labels?
    if isinstance(axs, plt.Axes):
        axs = [axs]
    elif isinstance(axs, np.ndarray):
        axs = axs.ravel()
    for ax in axs:
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=x_label_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=y_label_fontsize)
        if title:
            ax.set_title(title, fontsize = title_fontsize)
        for label in ax.get_xticklabels():
            label.set_rotation(x_label_rotation)
            label.set_fontsize(x_tick_fontsize)
            label.set_ha(x_label_ha)
            label.set_rotation_mode(x_label_rotationmode)
        for label in ax.get_yticklabels():
            label.set_fontsize(y_tick_fontsize)
            label.set_rotation(y_label_rotation)
    if tight_rect:
        fig.tight_layout(rect=tight_rect)
    else:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    if not show and not save_path:
        return fig, axs


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
