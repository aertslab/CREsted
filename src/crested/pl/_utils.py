"""Utility functions for plotting in CREsted."""

from __future__ import annotations

import matplotlib.pyplot as plt


def render_plot(
    fig,
    width: int = 8,
    height: int = 8,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title_fontsize: int = 16,
    x_label_fontsize: int = 14,
    y_label_fontsize: int = 14,
    x_tick_fontsize: int = 12,
    y_tick_fontsize: int = 12,
    x_label_rotation: int = 0,
    y_label_rotation: int = 0,
    show: bool = True,
    save_path: str | None = None,
) -> None | plt.Figure:
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
    show
        Whether to display the plot. Set this to False if you want to return the figure object to customize it further.
    save_path
        Optional path to save the figure. If None, the figure is displayed but not saved.
    """
    fig.set_size_inches(width, height)
    if title:
        fig.suptitle(title, fontsize=title_fontsize)
    for ax in fig.axes:
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=x_label_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=y_label_fontsize)
        for label in ax.get_xticklabels():
            label.set_rotation(x_label_rotation)
            label.set_fontsize(x_tick_fontsize)
        for label in ax.get_yticklabels():
            label.set_fontsize(y_tick_fontsize)
            label.set_rotation(y_label_rotation)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if not show and not save_path:
        return fig
