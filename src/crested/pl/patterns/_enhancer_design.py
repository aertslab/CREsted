from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from crested.pl._utils import render_plot

from ._utils import (
    _plot_attribution_map,
    _plot_mutagenesis_map,
    grad_times_input_to_df,
    grad_times_input_to_df_mutagenesis,
)


def enhancer_design_steps_contribution_score(
    intermediate,
    scores_all: np.ndarray,
    seqs_one_hot_all: np.ndarray,
    labels: list | None = None,
    zoom_n_bases: int | None = None,
    highlight_positions: list[tuple[int, int]] | None = None,
    ylim: tuple | None = None,
    method: str | None = None,
    **kwargs,
):
    if not isinstance(scores_all, list):
        scores_all = [scores_all]
    if not isinstance(seqs_one_hot_all, list):
        seqs_one_hot_all = [seqs_one_hot_all]

    # Plot
    for intermediate_idx, (scores, seqs_one_hot) in enumerate(
        zip(scores_all, seqs_one_hot_all)
    ):
        # Center and zoom
        if zoom_n_bases is None:
            zoom_n_bases = scores.shape[2]
        if labels and not isinstance(labels, list):
            labels = [str(labels)]
        center = int(scores.shape[2] / 2)
        start_idx = center - int(zoom_n_bases / 2)
        scores = scores[:, :, start_idx : start_idx + zoom_n_bases, :]
        intermediate_current = intermediate[intermediate_idx]
        for class_idx in range(scores.shape[1]):
            logger.info(
                f"Plotting contribution scores for {seqs_one_hot.shape[0]} sequence(s)"
            )
            number_of_steps = seqs_one_hot.shape[0]
            fig, ax = plt.subplots(
                number_of_steps, 1, figsize=(50, 2 * number_of_steps)
            )
            for seq in range(seqs_one_hot.shape[0]):
                seq_class_x = seqs_one_hot[seq, start_idx : start_idx + zoom_n_bases, :]
                if method == "mutagenesis":
                    global_max = scores[seq].max() + 0.25 * np.abs(scores[seq].max())
                    global_min = scores[seq].min() - 0.25 * np.abs(scores[seq].min())
                else:
                    mins = []
                    maxs = []
                    for i in range(scores.shape[1]):
                        seq_class_scores = scores[seq, i, :, :]
                        mins.append(np.min(seq_class_scores * seq_class_x))
                        maxs.append(np.max(seq_class_scores * seq_class_x))
                    global_max = np.array(maxs).max() + 0.25 * np.abs(
                        np.array(maxs).max()
                    )
                    global_min = np.array(mins).min() - 0.25 * np.abs(
                        np.array(mins).min()
                    )

                seq_class_scores = scores[seq, class_idx, :, :]
                if method == "mutagenesis":
                    mutagenesis_df = grad_times_input_to_df_mutagenesis(
                        seq_class_x, seq_class_scores
                    )
                    _plot_mutagenesis_map(mutagenesis_df, ax=ax[seq])
                else:
                    intgrad_df = grad_times_input_to_df(seq_class_x, seq_class_scores)
                    _plot_attribution_map(intgrad_df, ax=ax[seq], return_ax=False)

                if ylim:
                    ax[seq].set_ylim(ylim[0], ylim[1])
                else:
                    ax[seq].set_ylim([global_min, global_max])

                # Draw rectangles to highlight positions
                change_loc, change = intermediate_current["changes"][seq]
                if change_loc != -1:
                    start, end = change_loc, change_loc + len(change)
                    ax[seq].add_patch(
                        plt.Rectangle(
                            (
                                start - start_idx - 0.5,
                                global_min,
                            ),
                            end - start,
                            global_max - global_min,
                            edgecolor="red",
                            facecolor="none",
                            linewidth=0.5,
                        )
                    )
            if labels:
                class_name = labels[class_idx]
            else:
                class_name = f"Class {class_idx}"
            kwargs["title"] = class_name
            plt.xticks(np.arange(0, zoom_n_bases, 50))

            if "width" not in kwargs:
                kwargs["width"] = 50
            if "height" not in kwargs:
                kwargs["height"] = 2 * number_of_steps
            if "supxlabel" not in kwargs:
                kwargs["supxlabel"] = "Position"
            if "supylabel" not in kwargs:
                kwargs["supylabel"] = "Scores"
            if "tight_rect" not in kwargs:
                kwargs["tight_rect"] = (0.02, 0.02, 1, 0.98)
            render_plot(fig, **kwargs)


def enhancer_design_steps_predictions():
    # TODO implement
    pass
