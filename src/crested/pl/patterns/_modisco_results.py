from __future__ import annotations

import os

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

from crested.pl._utils import create_plot, render_plot
from crested.tl.modisco._modisco_utils import (
    _pattern_to_ppm,
    _trim_pattern_by_ic,
    compute_ic,
)
from crested.utils._logging import log_and_raise

from ._utils import _plot_attribution_map


def modisco_results(
    classes: list[str],
    contribution: str,
    contribution_dir: str,
    num_seq: int,
    viz: str = "contrib",
    min_seqlets: int = 0,
    verbose: bool = False,
    y_min: float = -0.05,
    y_max: float = 0.25,
    background: list[float] = None,
    trim_pattern: bool = True,
    trim_ic_threshold: float = 0.1,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Plot genomic contributions for the given classes.

    Requires the modisco results to be present in the specified directory.
    The contribution scores are trimmed based on information content (IC).

    Parameters
    ----------
    classes
        List of classes to plot genomic contributions for.
    contribution
        Contribution type to plot. Choose either "positive" or "negative".
    contribution_dir
        Directory containing the modisco results.
        Each class should have a separate modisco .h5 results file in the format {class}_modisco_results.h5.
    num_seq
        Total number of sequences used for the modisco run.
        Necessary to calculate the percentage of sequences with the pattern.
    viz
        Visualization method. Choose either "contrib" or "pwm".
    min_seqlets
        Minimum number of seqlets required for a pattern to be considered.
    verbose
        Print verbose output.
    y_min
        Minimum y-axis limit for the plot if viz is "contrib".
    y_max
        Maximum y-axis limit for the plot if viz is "contrib".
    background
        Background probabilities for each nucleotide. Default is [0.27, 0.23, 0.23, 0.27].
    trim_pattern
        Boolean for trimming modisco patterns.
    trim_ic_threshold
        If trimming patterns, indicate threshold.
    width
        Width of the newly created figure. Default is 6*`len(classes)`.
    height
        Height of the newly created figure. Default is 2*`max_num_patterns`.
    sharex
        Whether to share the x axes of the created plots. Default is False.
    sharey
        Whether to share the y axes of the created plots. Default is False.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.

    See Also
    --------
    crested.tl.modisco.tfmodisco
    crested.pl.render_plot

    Examples
    --------
    >>> crested.pl.patterns.modisco_results(
    ...     classes=["Lamp5", "Pvalb", "Sst", ""Sst-Chodl", "Vip"],
    ...     contribution="positive",
    ...     contribution_dir="/path/to/modisco_results",
    ...     num_seq=1000,
    ...     viz="pwm",
    ...     save_path="/path/to/genomic_contributions.png",
    ... )

    .. image:: ../../../../docs/_static/img/examples/genomic_contributions.png
    """

    @log_and_raise(ValueError)
    def _check_input_params():
        if contribution not in ["positive", "negative"]:
            raise ValueError(
                "Argument 'contribution' must be either 'positive' or 'negative'."
            )
        if viz not in ["contrib", "pwm"]:
            raise ValueError("Argument 'viz' must be either' 'contrib' or 'pwm'.")

    @log_and_raise(FileNotFoundError)
    def _check_file_params():
        if not os.path.isdir(contribution_dir):
            raise FileNotFoundError(
                f"contribution_dir {contribution_dir} is not a directory."
            )
        for class_name in classes:
            if not os.path.exists(
                os.path.join(contribution_dir, f"{class_name}_modisco_results.h5")
            ):
                raise FileNotFoundError(
                    f"Could not find modisco output file {class_name}_modisco_results.h5 in contribution_dir {contribution_dir}."
                )

    _check_input_params()
    _check_file_params()

    if background is None:
        background = [0.27, 0.23, 0.23, 0.27]
    background = np.array(background)
    pos_pat = contribution == "positive"

    logger.info(f"Starting genomic contributions plot for classes: {classes}")

    max_num_patterns = 0
    all_patterns = []

    for cell_type in classes:
        if verbose:
            logger.info(cell_type)
        hdf5_results = h5py.File(
            os.path.join(contribution_dir, f"{cell_type}_modisco_results.h5"), "r"
        )
        metacluster_names = list(hdf5_results.keys())

        if f"{contribution[:3]}_patterns" not in metacluster_names:
            raise ValueError(
                f"No {contribution[:3]}_patterns for {cell_type}. Aborting..."
            )

        for metacluster_name in [f"{contribution[:3]}_patterns"]:
            all_pattern_names = list(hdf5_results[metacluster_name])
            max_num_patterns = max(max_num_patterns, len(all_pattern_names))

    fig, axes = create_plot(
        ax=None,
        kwargs_dict=kwargs,
        nrows=max_num_patterns,
        ncols=len(classes),
        default_width=6*len(classes),
        default_height=2*max_num_patterns
    )

    if verbose:
        logger.info(f"Max patterns for selected classes: {max_num_patterns}")

    motif_counter = 1

    for idx, cell_type in enumerate(classes):
        hdf5_results = h5py.File(
            os.path.join(contribution_dir, f"{cell_type}_modisco_results.h5"), "r"
        )
        metacluster_names = list(hdf5_results.keys())

        if verbose:
            logger.info(metacluster_names)

        for metacluster_name in [f"{contribution[:3]}_patterns"]:
            all_pattern_names = list(hdf5_results[metacluster_name])

            for i in range(len(all_pattern_names)):
                pattern_name = "pattern_" + str(i)
                if len(classes) > 1:
                    ax = axes[motif_counter - 1, idx]
                elif max_num_patterns > 1:
                    ax = axes[motif_counter - 1]
                else:
                    ax = axes
                motif_counter += 1
                all_patterns.append((metacluster_name, pattern_name))
                pattern = hdf5_results[metacluster_name][pattern_name]
                num_seqlets = list(
                    hdf5_results[metacluster_name][pattern_name]["seqlets"]["n_seqlets"]
                )[0]
                if verbose:
                    logger.info(metacluster_name, pattern_name)
                    logger.info("total seqlets:", num_seqlets)
                if num_seqlets < min_seqlets:
                    break
                pattern_trimmed = (
                    _trim_pattern_by_ic(pattern, pos_pat, trim_ic_threshold)
                    if trim_pattern
                    else pattern
                )
                if viz == "contrib":
                    ax = _plot_attribution_map(
                        ax=ax,
                        saliency_df=np.array(pattern_trimmed["contrib_scores"]),
                        return_ax=True,
                        figsize=None,
                    )
                    ax.set_ylim([y_min, y_max])
                    ax.set_title(
                        f"{cell_type}: {np.around(num_seqlets / num_seq * 100, 2)}% seqlet frequency"
                    )
                elif viz == "pwm":
                    ppm = _pattern_to_ppm(pattern_trimmed)
                    ic, ic_pos, ic_mat = compute_ic(ppm)
                    pwm = np.array(ic_mat)
                    rounded_mean = np.around(np.mean(pwm), 2)
                    pwm = pwm.astype(float)
                    ax = _plot_attribution_map(
                        ax=ax, saliency_df=pwm, return_ax=True, figsize=None
                    )
                    ax.set_title(
                        f"{cell_type}: {np.around(num_seqlets / num_seq * 100, 2)}% seqlet frequency - Average IC: {rounded_mean:.2f}"
                    )
                    ax.set_ylim([0, 2])
                else:
                    raise ValueError(
                        'Invalid visualization method. Choose either "contrib" or "pwm". Aborting...'
                    )
        motif_counter = 1

    return render_plot(fig, axes, **kwargs)


def clustermap_tomtom_similarities(
    sim_matrix: np.ndarray,
    ids: list[str],
    pattern_dict: dict[str, dict],
    group_info: list[tuple[list[str], dict[str, str]]] = None,
    query_id: str | None = None,
    threshold: float | None = None,
    min_seqlets: int = 0,
    class_names: list[str] | None = None,
    cmap: str | plt.Colormap  = 'viridis',
    dendrogram_ratio: tuple[float, float] = (0.05, 0.05),
    logo_width_fraction: float = 0.35,
    logo_x_padding: float = 0.5,
    show_pwms: bool = True,
    save_path: str | None = None,
    width: int | float = 10,
    height: int | float = 10,
    plot_kws: dict | None = None,
    figsize = 'deprecated'
) -> sns.matrix.ClusterGrid:
    """
    Create a Seaborn clustermap of TOMTOM similarity scores with optional PWM logo display and filtering.

    Parameters
    ----------
    sim_matrix
        2D square array of TOMTOM similarity scores (-log10 p-values), shape (N, N).
    ids
        List of pattern identifiers corresponding to rows/columns of sim_matrix.
    pattern_dict
        Dictionary mapping pattern IDs to metadata. Each entry should contain:
        - 'n_seqlets': number of seqlets contributing to the pattern.
        - 'contrib_scores': DataFrame or array used for PWM logo plotting.
    group_info
        List of (group_labels, color_map) tuples. Each group_labels list has the same length as ids,
        and each color_map assigns colors to group values.
    query_id
        If provided, only show motifs with similarity > `threshold` to this ID.
    threshold
        Minimum TOMTOM score for similarity filtering (used only with `query_id`).
    min_seqlets
        Minimum number of seqlets required for a pattern to be shown.
    class_names
        If provided, only keep patterns whose class name (parsed as '_'.join(id.split('_')[:-3]))
        is in this list.
    cmap
        Colormap to use in the clustermap.
    dendrogram_ratio : tuple[float, float]
        Ratio of dendrogram size to figure size for rows and columns.
    logo_width_fraction
        Width of the PWM logo strip relative to the heatmap width.
    logo_x_padding
        Horizontal space between the PWM logos and the heatmap.
    show_pwms
        Whether to display PWM logos to the left of the heatmap.
    width
        Figure width.
    height
        Figure height.
    save_path
        If provided, the figure is saved to this path (e.g., as a PNG or PDF).
    plot_kws
        Extra keyword arguments passed to :func:`~seaborn.clustermap`.

    Returns
    -------
    sns.matrix.ClusterGrid
        The Seaborn clustermap object containing the heatmap and dendrograms.
    """
    if figsize != 'deprecated':
        width, height = figsize
        logger.warning("Argument `figsize` is deprecated, please use width and height instead.")
    if group_info is None:
        group_info = []

    ids_arr = np.array(ids)

    # --- Step 0: Filter by min_seqlets and class_names ---
    ids_filtered = []
    for i in ids_arr:
        ct = "_".join(i.split("_")[:-3])
        if pattern_dict[i]["n_seqlets"] >= min_seqlets and (
            class_names is None or ct in class_names
        ):
            ids_filtered.append(i)

    keep_idx = [i for i, x in enumerate(ids_arr) if x in ids_filtered]
    sim_matrix = sim_matrix[np.ix_(keep_idx, keep_idx)]
    ids_arr = ids_arr[keep_idx]
    group_info = [([g[i] for i in keep_idx], colors) for g, colors in group_info]

    # --- Step 1: Optional similarity filter ---
    if query_id is not None and threshold is not None:
        assert query_id in ids_arr, f"{query_id} not in filtered IDs"
        idx = np.where(ids_arr == query_id)[0][0]
        scores = sim_matrix[idx]
        keep_indices = np.where((scores > threshold) & (np.arange(len(scores)) != idx))[
            0
        ]
        keep_indices = np.append(keep_indices, idx)

        sim_matrix = sim_matrix[np.ix_(keep_indices, keep_indices)]
        ids_arr = ids_arr[keep_indices]
        group_info = [
            ([g[i] for i in keep_indices], colors) for g, colors in group_info
        ]

    # --- Step 2: Build row_colors matrix ---
    row_colors = []
    try:
        for group_labels, group_colors in group_info:
            row_colors.append([group_colors[g] for g in group_labels])
    except KeyError as e:
        print("!! KeyError in color mapping:", e)
        raise

    row_colors = np.array(row_colors)  # shape (n_groups, n_rows)

    # --- Step 3: Clustermap ---
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    g = sns.clustermap(
        sim_matrix,
        xticklabels=ids_arr,
        yticklabels=ids_arr,
        row_colors=row_colors,
        col_colors=row_colors,
        figsize=(width, height),
        cmap = cmap,
        dendrogram_ratio=dendrogram_ratio,
        **plot_kws
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90)

    # --- Step 4: PWM logos ---
    if show_pwms:
        row_order = g.dendrogram_row.reordered_ind
        reordered_ids = [ids_arr[i] for i in row_order]

        fig = g.fig
        extra_width = logo_width_fraction * width
        fig.set_size_inches(width + extra_width, height)

        heatmap_pos = g.ax_heatmap.get_position()
        logo_width = logo_width_fraction * heatmap_pos.width
        logo_height = heatmap_pos.height / len(reordered_ids)

        for i, motif_id in enumerate(reordered_ids):
            y_start = heatmap_pos.y0 + (len(reordered_ids) - 1 - i) * logo_height
            x_start = heatmap_pos.x0 - logo_width - logo_width * logo_x_padding
            ppm = pattern_dict[motif_id]["contrib_scores"]

            pwm_ax = fig.add_axes([x_start, y_start, logo_width, logo_height])
            _plot_attribution_map(
                ax=pwm_ax,
                saliency_df=ppm,
                return_ax=True,
                figsize=(8, 2),
                rotate=False,
            )
            pwm_ax.axis("off")

    # --- Step 5: Move colorbar out of the way ---
    heatmap_pos = g.ax_heatmap.get_position()
    cbar = g.cax
    cbar_height = heatmap_pos.height / 4
    cbar_width = heatmap_pos.width / 20
    cbar_x = heatmap_pos.x1 + 0.02
    cbar_y = heatmap_pos.y0 - 0.2
    cbar.set_position([cbar_x, cbar_y, cbar_width, cbar_height])
    g.cax.set_ylabel("Motif similarity (−log₁₀ p-value)", rotation=270, labelpad=15)

    # --- Step 6: Add a legend for the first group ---
    if group_info:
        group_labels, group_colors = group_info[0]
        legend_elements = [
            Patch(facecolor=color, edgecolor="black", label=label)
            for label, color in group_colors.items()
        ]
        g.fig.legend(
            handles=legend_elements,
            title="Class",
            loc="lower left",
            bbox_to_anchor=(-0.06, 0.05),
            frameon=True,
            framealpha=0.9,
            borderpad=0.6,
            fontsize=11,
            title_fontsize=11,
        )

    if save_path is not None:
        g.savefig(save_path, bbox_inches="tight")

    return g


def clustermap(
    pattern_matrix: np.ndarray,
    classes: list[str],
    subset: list[str] | None = None,
    grid: bool = False,
    cmap: str = "coolwarm",
    center: float = 0,
    method: str = "average",
    dy: float = 0.002,
    save_path: str | None = None,
    pat_seqs: list[tuple[str, np.ndarray]] | None = None,
    dendrogram_ratio: tuple[float, float] = (0.05, 0.2),
    importance_threshold: float = 0,
    plot_kws: dict | None = None,
    width: int | float = 25,
    height: int | float = 8,
    figsize = 'deprecated',
) -> sns.matrix.ClusterGrid:
    """
    Create a clustermap from the given pattern matrix and class labels with customizable options.

    Parameters
    ----------
    pattern_matrix
        2D NumPy array containing pattern data.
    classes
        List of class labels, matching the rows of the pattern matrix.
    subset
        List of class labels to subset the matrix.
    grid
        Whether to add a grid to the heatmap.
    cmap
        Colormap for the clustermap.
    center
        Value at which to center the colormap.
    method
        Clustering method to use.
    dy
        Scaling parameter for vertical distance between nucleotides (if pat_seqs is not None) in xticklabels.
    save_path
        Path to save the figure.
    pat_seqs
        List of sequences to use as xticklabels.
    dendrogram_ratio
        Ratio of dendograms in x and y directions.
    importance_threshold
        Minimal pattern importance threshold over all classes to retain the pattern before clustering and plotting.
    width
        Figure width.
    height
        Figure height.
    plot_kws
        Extra keyword arguments passed to :func:`~seaborn.clustermap`.
        Default is `{'cbar_pos': (1.05, 0.4, 0.01, 0.3)}`.

    See Also
    --------
    crested.tl.modisco.create_pattern_matrix
    crested.tl.modisco.generate_nucleotide_sequences

    Examples
    --------
    >>> pat_seqs = crested.tl.modisco.generate_nucleotide_sequences(all_patterns)
    >>> crested.pl.patterns.clustermap(
    ...     pattern_matrix,
    ...     classes=list(adata.obs_names)
    ...     subset=["Lamp5", "Pvalb", "Sst", "Sst-Chodl", "Vip"],
    ...     figsize=(25, 8),
    ...     grid=True,
    ... )

    .. image:: ../../../../docs/_static/img/examples/pattern_clustermap.png
    """
    if figsize != 'deprecated':
        width, height = figsize
        logger.warning("Argument `figsize` is deprecated, please use width and height instead.")
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'cbar_pos' not in plot_kws:
        plot_kws['cbar_pos'] = (1.05, 0.4, 0.01, 0.3)

    # Subset the pattern_matrix and classes if subset is provided
    if subset is not None:
        subset_indices = [
            i for i, class_label in enumerate(classes) if class_label in subset
        ]
        pattern_matrix = pattern_matrix[subset_indices, :]
        classes = [classes[i] for i in subset_indices]

    # Filter columns based on importance threshold
    max_importance = np.max(np.abs(pattern_matrix), axis=0)
    above_threshold = max_importance > importance_threshold
    pattern_matrix = pattern_matrix[:, above_threshold]

    if pat_seqs is not None:
        pat_seqs = [pat_seqs[i] for i in np.where(above_threshold)[0]]

    data = pd.DataFrame(pattern_matrix)

    if pat_seqs is not None:
        plt.rc("text", usetex=False)  # Turn off LaTeX to speed up rendering

    g = sns.clustermap(
        data,
        cmap=cmap,
        figsize=(width, height),
        yticklabels=classes,
        center=center,
        xticklabels=True if pat_seqs is None else False,
        method=method,
        dendrogram_ratio=dendrogram_ratio,
        **plot_kws,

    )
    col_order = g.dendrogram_col.reordered_ind
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.set_label(
        "Motif importance", rotation=270, labelpad=20
    )  # Rotate label and add padding
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # Get the reordered column indices from the clustermap
    col_order = g.dendrogram_col.reordered_ind

    # Reorder the pat_seqs to follow the column order
    if pat_seqs is not None:
        reordered_pat_seqs = [pat_seqs[i] for i in col_order]
        ax = g.ax_heatmap
        x_positions = (
            np.arange(len(reordered_pat_seqs)) + 0.5
        )  # Shift labels by half a tick to the right

        constant = (1 / height) * 64
        for i, (letters, scores) in enumerate(reordered_pat_seqs):
            previous_spacing = 0
            for _, (letter, score) in enumerate(
                zip(reversed(letters), reversed(scores), strict=False)
            ):
                fontsize = score * 10
                vertical_spacing = max(
                    (constant * score * dy), constant * 0.1 * dy
                )  # Spacing proportional to figsize[1]

                ax.text(
                    x_positions[i],
                    -(constant * 0.002)
                    - previous_spacing,  # Adjust y-position based on spacing
                    letter,
                    fontsize=fontsize,  # Constant font size
                    ha="center",  # Horizontal alignment
                    va="top",  # Vertical alignment
                    rotation=90,  # Rotate the labels vertically
                    transform=ax.get_xaxis_transform(),  # Ensure the text is placed relative to x-axis
                )
                previous_spacing += vertical_spacing

        # Ensure x-ticks are visible
        ax.set_xticks(x_positions)

    if grid:
        ax = g.ax_heatmap
        # Define the grid positions (between cells, hence the +0.5 offset)
        x_positions = np.arange(pattern_matrix.shape[1] + 1)
        y_positions = np.arange(len(pattern_matrix) + 1)

        # Add horizontal grid lines
        for y in y_positions:
            ax.hlines(y, *ax.get_xlim(), color="grey", linewidth=0.25)

        # Add vertical grid lines
        for x in x_positions:
            ax.vlines(x, *ax.get_ylim(), color="grey", linewidth=0.25)

        g.fig.canvas.draw()

    if save_path is not None:
        g.savefig(save_path, bbox_inches="tight")

    return g

def clustermap_with_pwm_logos(
    pattern_matrix: np.ndarray,
    classes: list[str],
    pattern_dict: dict,
    subset: list[str] | None = None,
    grid: bool = False,
    cmap: str = "coolwarm",
    center: float = 0,
    method: str = "average",
    save_path: str | None = None,
    dendrogram_ratio: tuple[float, float] = (0.05, 0.2),
    importance_threshold: float = 0,
    logo_height_fraction: float = 0.35,
    logo_y_padding: float = 0.3,
    pwm_or_contrib: str = "pwm",
    width: int | float = 25,
    height: int | float = 8,
    plot_kws: dict | None = None,
    figsize = 'deprecated'
) -> sns.matrix.ClusterGrid:
    """
    Create a clustermap with additional PWM logo plots below the heatmap.

    Parameters
    ----------
    pattern_matrix:
        A 2D array representing the data matrix for clustering.
    classes:
        The class labels for the rows of the matrix.
    pattern_dict:
        A dictionary containing PWM patterns for x-tick plots.
    subset
        List of class labels to subset the matrix.
    grid:
        Whether to overlay grid lines on the heatmap. Default is False.
    cmap:
        Colormap for the heatmap. Default is "coolwarm".
    center:
        The value at which to center the colormap. Default is 0.
    method:
        Linkage method for hierarchical clustering. Default is "average".
    save_path:
        Path to save the final figure. If None, the figure is not saved. Default is None.
    dendrogram_ratio:
        Ratios for the size of row and column dendrograms. Default is (0.05, 0.2).
    importance_threshold:
        Threshold for filtering columns based on maximum absolute importance. Default is 0.
    logo_height_fraction:
        Fraction of clustermap height to allocate for PWM logos. Default is 0.35.
    logo_y_padding:
        Relative vertical padding for the PWM logos relative to the heatmap. Default is 0.3.
    pwm_or_contrib:
        Whether to use the pwm or contrib score representation of the pattern in the plotting.
    width
        Width of the newly created figure. Default is 25.
    height
        Height of the newly created figure. Default is 8.
    plot_kws
        Extra keyword arguments passed to :func:`~seaborn.clustermap`.
        Default is `{'cbar_pos': (1.05, 0.4, 0.01, 0.3)}`.

    Returns
    -------
        sns.matrix.ClusterGrid: A seaborn ClusterGrid object containing the clustermap with the PWM logos.
    """
    if figsize != 'deprecated':
        width, height = figsize
        logger.warning("Argument `figsize` is deprecated, please use width and height instead.")
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'cbar_pos' not in plot_kws:
        plot_kws['cbar_pos'] = (1.05, 0.4, 0.01, 0.3)
    # Subset the pattern_matrix and classes if subset is provided
    if subset is not None:
        subset_indices = [
            i for i, class_label in enumerate(classes) if class_label in subset
        ]
        pattern_matrix = pattern_matrix[subset_indices, :]
        classes = [classes[i] for i in subset_indices]

    # Filter columns based on importance threshold
    max_importance = np.max(np.abs(pattern_matrix), axis=0)
    above_threshold = max_importance > importance_threshold
    pattern_matrix = pattern_matrix[:, above_threshold]

    # Subset the pattern_dict to match filtered columns
    selected_patterns = [pattern_dict[str(i)] for i in np.where(above_threshold)[0]]
    selected_indices = list(np.where(above_threshold)[0])

    data = pd.DataFrame(pattern_matrix)

    # Generate the clustermap with the specified figsize
    g = sns.clustermap(
        data,
        cmap=cmap,
        figsize=(width, height),
        yticklabels=classes,
        center=center,
        xticklabels=False,
        method=method,
        dendrogram_ratio=dendrogram_ratio,
        **plot_kws
    )

    col_order = g.dendrogram_col.reordered_ind
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.set_label("Motif importance", rotation=270, labelpad=20)

    # Reorder selected_patterns based on clustering
    reordered_patterns = [selected_patterns[i] for i in col_order]
    reordered_indices = [selected_indices[i] for i in col_order]

    # Compute space for x-tick images
    extra_height = logo_height_fraction * height
    total_height = height + extra_height

    # Update the figure size to accommodate the logos
    fig = g.fig
    fig.set_size_inches(width, total_height)

    # Adjust width and height of logos
    logo_width = g.ax_heatmap.get_position().width / len(reordered_patterns)
    logo_height = logo_height_fraction * g.ax_heatmap.get_position().height
    ratio = logo_height / logo_width

    for i, pattern in enumerate(reordered_patterns):
        plot_start_x = (
            g.ax_heatmap.get_position().x0
            + g.ax_heatmap.get_position().width / len(reordered_patterns) * i
        )
        plot_start_y = (
            g.ax_heatmap.get_position().y0 - logo_height * (1+logo_y_padding)
        )
        pwm_ax = fig.add_axes([plot_start_x, plot_start_y, logo_width, logo_height])
        pwm_ax.clear()

        pwm = None
        if pwm_or_contrib == "pwm":
            ppm = _pattern_to_ppm(pattern["pattern"])
            ic, ic_pos, ic_mat = compute_ic(ppm)
            pwm = np.array(ic_mat)
        elif pwm_or_contrib == "contrib":
            pwm = np.array(pattern["pattern"]["contrib_scores"])
        else:
            raise ValueError(
                'Invalid visualization method. Choose either "contrib" or "pwm" in the pwm_or_contrib parameter. Aborting...'
            )

        pwm_ax = _plot_attribution_map(
            ax=pwm_ax,
            saliency_df=pwm,
            return_ax=True,
            figsize=(8 * ratio, 8),
            rotate=True,
        )
        pwm_ax.axis("off")

    if grid:
        ax = g.ax_heatmap
        x_positions = np.arange(pattern_matrix.shape[1] + 1)
        y_positions = np.arange(len(pattern_matrix) + 1)

        # Add horizontal grid lines
        for y in y_positions:
            ax.hlines(y, *ax.get_xlim(), color="grey", linewidth=0.25)

        # Add vertical grid lines
        for x in x_positions:
            ax.vlines(x, *ax.get_ylim(), color="grey", linewidth=0.25)

        g.fig.canvas.draw()

    ax = g.ax_heatmap
    ax.xaxis.tick_bottom()
    ax.set_xticks(np.arange(pattern_matrix.shape[1]) + 0.5)
    ax.set_xticklabels([str(i) for i in reordered_indices], rotation=90)
    for tick in ax.get_xticklabels():
        tick.set_verticalalignment("top")

    if save_path is not None:
        g.savefig(save_path, bbox_inches="tight", dpi=600)

    return g


def selected_instances(
    pattern_dict: dict,
    idcs: list[int],
    ax: plt.Axes | None = None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes] | tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Plot the patterns specified by the indices in `idcs` from the `pattern_dict`.

    Parameters
    ----------
    pattern_dict
        A dictionary containing pattern data. Each key corresponds to a pattern ID, and the value is a nested structure containing
        contribution scores and metadata for the pattern. Refer to the output of `crested.tl.modisco.process_patterns`.
    idcs
        An index or list of indices specifying which patterns to plot. The indices correspond to keys in the `pattern_dict`.
    ax
        An axis to plot the instance on. Only works if providing a single value for idcs.
    width
        Width of the newly created figure if `ax=None`. Default is 8.
    height
        Height of the newly created figure if `ax=None`. Default is 2*`len(idcs)`.
    sharex
        Whether to share the x axes of the created plots. Default is False.
    sharey
        Whether to share the y axes of the created plots. Default is False.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Defaults for `selected_instances`: `title=[pattern_dict[str(idx)]["pattern"]["id"] for idx in idcs]`.

    See Also
    --------
    crested.tl.modisco.process_patterns

    Examples
    --------
    >>> pattern_indices = [0, 1, 2]
    >>> crested.pl.patterns.selected_instances(pattern_dict, pattern_indices)

    .. image:: ../../../../docs/_static/img/examples/pattern_selected_instances.png
    """
    if isinstance(idcs, str):
        idcs = [idcs]
    if len(idcs) > 1 and ax is not None:
        raise ValueError("Can only provide a pre-existing axis if plotting a single index at idcs.")

    # Create figure
    fig, axs = create_plot(
        ax=ax,
        kwargs_dict=kwargs,
        nrows=len(idcs),
        ncols=1,
        default_width=8,
        default_height=2*len(idcs),
        default_sharex=False,
        default_sharey=False,
    )
    if len(idcs) == 1:
        axs = [axs]
    if 'title' not in kwargs:
        kwargs['title'] = [pattern_dict[str(idx)]["pattern"]["id"] for idx in idcs]

    for i, idx in enumerate(idcs):
        _plot_attribution_map(
            ax=axs[i],
            saliency_df=np.array(pattern_dict[str(idx)]["pattern"]["contrib_scores"]),
            return_ax=False,
            figsize=None,
        )

    return render_plot(fig, axs, **kwargs)


def class_instances(
    pattern_dict: dict,
    idx: int,
    class_representative: bool = False,
    **kwargs
) -> tuple[plt.Figure, list[plt.Axes]] |  None:
    """
    Plot instances of a specific pattern, either the representative pattern per class or all instances for a given pattern index.

    Parameters
    ----------
    pattern_dict
        A dictionary containing pattern data. Each key corresponds to a pattern ID, and each value contains instances of the pattern
        across different classes, along with their contribution scores. Refer to the output of `crested.tl.modisco.process_patterns`.
    idx
        The index specifying which pattern's instances to plot. This corresponds to a key in the `pattern_dict`.
    class_representative
        If True, only the best representative instance of each class is plotted. If False (default), all instances of the pattern
        within each class are plotted.
    width
        Width of the newly created figure. Default is 8.
    height
        Height of the newly created figure. Default is 2*len(idcs).
    sharex
        Whether to share the x axes of the created plots. Default is False.
    sharey
        Whether to share the y axes of the created plots. Default is False.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Defaults for `class_instances`: `title=[pattern_dict[str(idx)]["pattern"]["id"] for idx in idcs]`.

    See Also
    --------
    crested.tl.modisco.process_patterns

    Examples
    --------
    >>> crested.pl.patterns.class_instances(pattern_dict, 0, class_representative=False)

    .. image:: ../../../../docs/_static/img/examples/pattern_class_instances.png
    """
    if class_representative:
        key = "classes"
    else:
        key = "instances"
    n_instances = len(pattern_dict[str(idx)][key])

    # Set defaults
    fig, axs = create_plot(
        ax=None,
        kwargs_dict=kwargs,
        nrows=n_instances,
        default_width=8,
        default_height=2*n_instances,
    )
    if n_instances == 1:
        axs = [axs]

    instance_classes = list(pattern_dict[str(idx)][key].keys())
    if 'title' not in kwargs:
        kwargs['title'] = [pattern_dict[str(idx)][key][cl]["id"] for cl in instance_classes]

    for i, cl in enumerate(instance_classes):
        _plot_attribution_map(
            ax=axs[i],
            saliency_df=np.array(pattern_dict[str(idx)][key][cl]["contrib_scores"]),
            return_ax=False,
            figsize=None,
        )

    return render_plot(fig, axs, **kwargs)


def similarity_heatmap(
    similarity_matrix: np.ndarray,
    indices: list,
    cmap: str | plt.Colormap = 'coolwarm',
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    fig_size = 'deprecated',
    fig_path = 'deprecated',
    **kwargs
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Plot a similarity heatmap of all pattern indices.

    Parameters
    ----------
    similarity_matrix
        A 2D numpy array containing the similarity values.
    indices
        List of pattern indices.
    cmap
        Colormap to use.
    plot_kws
        Extra keyword arguments passed to :func:`~seaborn.heatmap`.
        Adjusted defaults compared to the base function are `{'annot': True, 'fmt': '.2f', 'annot_kws': {'size': 8}, 'xticklabels': indices, 'yticklabels': indices}`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default is 30.
    height
        Height of the newly created figure if `ax=None`. Default is 15.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Adjusted defaults for `similarity_heatmap`: `title="Pattern similarity heatmap"`, `title_fontsize=20`, `[x/y]label='Pattern index'`, `[x/y]label_fontsize=15`

    See Also
    --------
    crested.tl.modisco.calculate_similarity_matrix

    Examples
    --------
    >>> sim_matrix, indices = crested.tl.modisco.calculate_similarity_matrix(
    ...     all_patterns
    ... )
    >>> crested.pl.patterns.similarity_heatmap(sim_matrix, indices, fig_size=(42, 17))

    .. image:: ../../../../docs/_static/img/examples/pattern_similarity_heatmap.png
    """
    # Handle deprecated arguments
    if fig_size != 'deprecated':
        kwargs['width'], kwargs['height'] = fig_size
        logger.warning("`fig_size` is deprecated, please use arguments `width` and `height` instead.")
    if fig_path != 'deprecated':
        kwargs['save_path'] = fig_path
        logger.warning("`fig_path` is deprecated, please use arguments `save_path` instead.") # handled in render_plot

    # Set defaults
    if 'title' not in kwargs:
        kwargs['title'] = "Pattern similarity heatmap"
    if 'title_fontsize' not in kwargs:
        kwargs['title_fontsize'] = 20
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = 'Pattern index'
    if 'xlabel_fontsize' not in kwargs:
        kwargs['xlabel_fontsize'] = 15
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Pattern index'
    if 'ylabel_fontsize' not in kwargs:
        kwargs['ylabel_fontsize'] = 15
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'annot' not in plot_kws:
        plot_kws['annot'] = True
    if 'fmt' not in plot_kws:
        plot_kws['fmt'] = '.2f'
    if 'annot_kws' not in plot_kws:
        plot_kws['annot_kws'] = {'size': 8}
    if 'xticklabels' not in plot_kws:
        plot_kws['xticklabels'] = indices
    if 'yticklabels' not in plot_kws:
        plot_kws['yticklabels'] = indices

    fig, ax = create_plot(ax=ax, kwargs_dict=kwargs, default_width=30, default_height=15)
    heatmap = sns.heatmap(
        similarity_matrix,
        ax=ax,
        cmap=cmap,
        **plot_kws
    )

    for _, spine in heatmap.spines.items():
        spine.set_visible(True)
        spine.set_color("grey")
        spine.set_linewidth(0.5)

    return render_plot(fig, ax, **kwargs)


def tf_expression_per_cell_type(
    df: pd.DataFrame,
    tf_list: list,
    log_transform: bool = False,
    plot_kws: dict | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Plot the expression levels of specified transcription factors (TFs) per cell type.

    Parameters
    ----------
    df
        The DataFrame containing mean gene expression data per cell type.
    tf_list
        A list of transcription factors (TFs) to plot.
    log_transform
        Whether to log-transform the TF expression values.
    plot_kws
        Extra keyword arguments passed to :meth:`~pandas.DataFrame.plot`.
        Defaults: `{'width': 0.8}`.
    ax
        Axis to plot values on. If not supplied, creates a figure from scratch.
    width
        Width of the newly created figure if `ax=None`. Default is 12.
    height
        Height of the newly created figure if `ax=None`. Default is 5.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.
        Defaults for `tf_expression_per_cell_type`: `title="TF expression per cell type"`, `xlabel="Cell type"`, `ylabel="(Log1p-transformed) mean TF expression"`, `xtick_rotation=45`, `xtick_ha=55`.
    """
    # Check if all specified TFs are in the DataFrame
    missing_tfs = [tf for tf in tf_list if tf not in df.columns]
    if missing_tfs:
        raise ValueError(
            f"The following TFs are not found in the DataFrame: {missing_tfs}"
        )

    # Subset the DataFrame to include only the specified TFs
    tf_expression_df = df[tf_list]

    # Apply log transformation if specified
    if log_transform:
        tf_expression_df = np.log1p(tf_expression_df)

    # Set defaults
    if 'title' not in kwargs:
        kwargs['title'] = "TF expression per cell type"
    if 'xlabel' not in kwargs:
        kwargs['xlabel'] =  "Cell type"
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = "Mean TF expression"
        if log_transform:
            kwargs['ylabel'] = "Log1p-transformed " + kwargs['ylabel'].lower()
    if 'xtick_rotation' not in kwargs:
        kwargs['xtick_rotation'] = 45
    if 'xtick_ha' not in kwargs:
        kwargs['xtick_ha'] = 'right'
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    if 'width' not in plot_kws:
        plot_kws['width'] = 0.8

    # Plot the TF expression per cell type
    fig, ax = create_plot(ax=ax, kwargs_dict=kwargs, default_width=12, default_height=5)
    ax = tf_expression_df.plot(kind="bar", ax = ax, **plot_kws)
    ax.legend(title="Transcription factors")

    return render_plot(fig, ax, **kwargs)


def clustermap_tf_motif(
    data: np.ndarray,
    heatmap_dim: str = "gex",
    dot_dim: str = "contrib",
    class_labels: list[str] | None = None,
    subset_classes: list[str] | None = None,
    pattern_labels: list[str] | None = None,
    cluster_rows: bool = True,
    cluster_columns: bool = True,
    imshow_kws: dict | None = None,
    scatter_kws: dict | None = None,
    fig_size = 'deprecated',
    **kwargs
) -> tuple[plt.Figure, list[plt.Axes]] | None:
    """
    Generate a heatmap where one modality is represented as color, and the other as dot size.

    Parameters
    ----------
    data
        3D numpy array with shape (len(classes), #patterns, 2).
    heatmap_dim
        Either 'gex' or 'contrib', indicating which third dimension to use for heatmap colors.
    dot_dim
        Either 'gex' or 'contrib', indicating which third dimension to use for dot sizes.
    class_labels
        Labels for the classes.
    subset_classes
        Subset of classes to include in the heatmap. Rows in `data` are filtered accordingly.
    pattern_labels
        Labels for the patterns.
    cluster_rows
        Whether to cluster the rows (classes). Default is True.
    cluster_columns
        Whether to cluster the columns (patterns). Default is True.
    imshow_kws
        Extra arguments for `ax.imshow`. Default is `{'cmap': 'coolwarm', 'aspect': 'auto'}`.
    scatter_kws
        Extra arguments for `ax.scatter`. Default is `{'c': "black", 'alpha': 0.6, 'edgecolor': "none"}`
    width
        Width of the newly created figure. Default is `max(20, data.shape[1]//4)`.
    height
        Height of the newly created figure. Default is `data.shape[0]//2`.
    kwargs
        Additional arguments passed to :func:`~crested.pl.render_plot` to control the final plot output.
        Please see :func:`~crested.pl.render_plot` for details.

    Examples
    --------
    >>> clustermap_tf_motif(
    ...     data,
    ...     heatmap_dim="gex",
    ...     dot_dim="contrib",
    ...     class_labels=classes,
    ...     pattern_labels=patterns,
    ...     cluster_rows=True,
    ...     cluster_columns=True,
    ... )
    """
    if fig_size != 'deprecated':
        kwargs['width'], kwargs['height'] = fig_size
        logger.warning("`fig_size` is deprecated, please use `width` and `height` instead.")
    assert data.shape[2] == 2, "The third dimension of the data must be 2."

    # Set default labels if not provided
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(data.shape[0])]
    if pattern_labels is None:
        pattern_labels = [f"Pattern {i}" for i in range(data.shape[1])]

    # Subset classes if specified
    if subset_classes is not None:
        subset_indices = [
            i for i, label in enumerate(class_labels) if label in subset_classes
        ]
        if not subset_indices:
            raise ValueError("No matching classes found in class_labels.")
        data = data[subset_indices, :, :]
        class_labels = [class_labels[i] for i in subset_indices]

    # Remove empty columns (columns with all zeros in dot_dim)
    non_empty_columns = np.any(data[:, :, 1] != 0, axis=0)

    # Further filter: keep columns where max value of the 0th dimension exceeds 0.5
    valid_columns = np.max(data[:, :, 0], axis=0) > 0.5
    valid_columns = np.logical_and(non_empty_columns, valid_columns)

    # Apply filtering to data and pattern labels
    data = data[:, valid_columns, :]
    pattern_labels = [
        label for i, label in enumerate(pattern_labels) if valid_columns[i]
    ]

    # Mapping for dimensions
    dim_mapping = {"gex": 0, "contrib": 1}
    heatmap_idx = dim_mapping[heatmap_dim]
    dot_idx = dim_mapping[dot_dim]

    # Optional clustering for rows
    if cluster_rows:
        row_clustering_data = data[:, :, dot_idx]
        row_linkage = linkage(row_clustering_data, method="ward")
        row_order = leaves_list(row_linkage)
        data = data[row_order, :, :]
        class_labels = [class_labels[i] for i in row_order]
    else:
        row_order = None

    # Optional clustering for columns
    if cluster_columns:
        col_clustering_data = data[:, :, dot_idx].T
        col_linkage = linkage(col_clustering_data, method="ward")
        col_order = leaves_list(col_linkage)
        data = data[:, col_order, :]
        pattern_labels = [pattern_labels[i] for i in col_order]
    else:
        col_order = None

    # Extract heatmap and dot size data
    heatmap_data = data[:, :, heatmap_idx]
    dot_size_data = data[:, :, dot_idx]
    max_dot_size = np.max(dot_size_data)
    dot_size_data = dot_size_data / max_dot_size  # Normalize dot size

    # Define figure layout
    plot_width = kwargs.pop('width') if 'width' in kwargs else max(20, data.shape[1] // 4)
    plot_height = kwargs.pop('height') if 'height' in kwargs else data.shape[0] // 2
    fig = plt.figure(figsize=(plot_width, plot_height))

    if cluster_rows:
        gs = fig.add_gridspec(1, 2, width_ratios=[0.1, 4], wspace=0.02)
        ax_dendro = fig.add_subplot(gs[0, 0])
        dendrogram(row_linkage, orientation="left", no_labels=True, ax=ax_dendro)
        ax_dendro.invert_yaxis()
        ax_dendro.axis("off")
        ax_heatmap = fig.add_subplot(gs[0, 1])
    else:
        ax_heatmap = fig.add_subplot(111)

    # Normalize colors for heatmap
    norm = mcolors.TwoSlopeNorm(
        vmin=-max(np.abs(heatmap_data.min()), np.abs(heatmap_data.max())),
        vcenter=0,
        vmax=max(np.abs(heatmap_data.min()), np.abs(heatmap_data.max())),
    )

    # Plot heatmap
    imshow_kws = {} if imshow_kws is None else imshow_kws.copy()
    if 'aspect' not in imshow_kws:
        imshow_kws['aspect'] = 'auto'
    if 'cmap' not in imshow_kws:
        imshow_kws['cmap'] = 'coolwarm'
    heatmap = ax_heatmap.imshow(
        heatmap_data,
        norm=norm,
        **imshow_kws
    )

    # Overlay dots
    scatter_kws = {} if scatter_kws is None else scatter_kws.copy()
    if 'c' not in scatter_kws:
        scatter_kws['c'] = 'black'
    if 'alpha' not in scatter_kws:
        scatter_kws['alpha'] = 0.6
    if 'edgecolor' not in scatter_kws:
        scatter_kws['edgecolor'] = 'none'
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax_heatmap.scatter(
                j,
                i,
                s=dot_size_data[i, j] * 100,
                **scatter_kws
            )

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax_heatmap)
    label = (
        "Average pattern contribution score"
        if heatmap_dim == "contrib"
        else "Average TF expression, signed by activation/repression"
    )
    cbar.set_label(label)

    # Set axis labels and ticks
    ax_heatmap.set_xticks(np.arange(data.shape[1]))
    ax_heatmap.set_xticklabels(pattern_labels, rotation=90)
    ax_heatmap.set_yticks(np.arange(data.shape[0]))
    ax_heatmap.set_yticklabels(class_labels)

    ax_heatmap.yaxis.tick_right()

    # Final layout adjustments - provide only heatmap ax since other axes likely shouldn't be adjusted/labeled/etc through this.
    return render_plot(fig, ax_heatmap, **kwargs)
