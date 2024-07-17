"""Utility functions for plotting in CREsted."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt


def render_plot(
    fig,
    width: int = 8,
    height: int = 8,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    supxlabel: str | None = None,
    supylabel: str | None = None,
    tight_rect: tuple | None = None,
    save_path: str | None = None,
) -> None:
    """
    Render a plot with customization options.

    Note
    ----
    This function should never be called directly. Rather, the other plotting functions call this function.

    Parameters
    ----------
    fig
        The figure object to render.
    width
        Width of the plot (inches).
    height
        Height of the plot (inches).
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
    fig_path
        Optional path to save the figure. If None, the figure is displayed but not saved.
    """
    fig.set_size_inches(width, height)
    if title:
        fig.suptitle(title)
    if supxlabel:
        fig.supxlabel(supxlabel)
    if supylabel:
        fig.supylabel(supylabel)
    for ax in fig.axes:
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
    if tight_rect:
        plt.tight_layout(rect=tight_rect)
    else:
        plt.tight_layout()
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    plt.show()
    plt.close()
