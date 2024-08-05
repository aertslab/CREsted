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
    x_label_rotation: int = 0,
    show: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
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
    x_label_rotation
        Rotation of the X-axis labels in degrees.
    show
        Whether to display the plot.
    save_path
        Optional path to save the figure. If None, the figure is displayed but not saved.
    """
    fig.set_size_inches(width, height)
    if title:
        fig.suptitle(title)
    for ax in fig.axes:
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        for label in ax.get_xticklabels():
            label.set_rotation(x_label_rotation)

    plt.tight_layout()
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.close(fig)

    return fig
