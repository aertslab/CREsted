from __future__ import annotations

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

from crested.pl._utils import render_plot
from crested.tl.modisco._modisco_utils import (
    _pattern_to_ppm,
    _trim_pattern_by_ic,
    compute_ic,
)

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
    **kwargs,
) -> None:
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
    kwargs
        Additional keyword arguments for the plot.

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
            f"{contribution_dir}/{cell_type}_modisco_results.h5", "r"
        )
        metacluster_names = list(hdf5_results.keys())

        if f"{contribution[:3]}_patterns" not in metacluster_names:
            raise ValueError(
                f"No {contribution[:3]}_patterns for {cell_type}. Aborting..."
            )

        for metacluster_name in [f"{contribution[:3]}_patterns"]:
            all_pattern_names = list(hdf5_results[metacluster_name])
            max_num_patterns = max(max_num_patterns, len(all_pattern_names))

    fig, axes = plt.subplots(
        nrows=max_num_patterns,
        ncols=len(classes),
        figsize=(8 * len(classes), 2 * max_num_patterns),
    )

    if verbose:
        logger.info(f"Max patterns for selected classes: {max_num_patterns}")

    motif_counter = 1

    for idx, cell_type in enumerate(classes):
        hdf5_results = h5py.File(
            f"{contribution_dir}/{cell_type}_modisco_results.h5", "r"
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
                pattern_trimmed = _trim_pattern_by_ic(pattern, pos_pat, 0.1)
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
                    pattern = _trim_pattern_by_ic(pattern, pos_pat, 0.1)
                    ppm = _pattern_to_ppm(pattern)
                    ic, ic_pos, ic_mat = compute_ic(ppm)
                    pwm = np.array(ic_mat)
                    rounded_mean = np.around(np.mean(pwm), 2)
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

    plt.tight_layout()
    if "width" not in kwargs:
        kwargs["width"] = 6 * len(classes)
    if "height" not in kwargs:
        kwargs["height"] = 2 * max_num_patterns

    return render_plot(fig, **kwargs)


def clustermap(
    pattern_matrix: np.ndarray,
    classes: list[str],
    subset: list[str] | None = None,  # Subset option
    figsize: tuple[int, int] = (25, 8),
    grid: bool = False,
    cmap: str = "coolwarm",
    center: float = 0,
    method: str = "average",
    dy: float = 0.002,
    fig_path: str | None = None,
    pat_seqs: list[tuple[str, np.ndarray]] | None = None,
    dendrogram_ratio: tuple[float, float] = (0.05, 0.2),
    importance_threshold : float = 0,
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
    figsize
        Size of the figure.
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
    fig_path
        Path to save the figure.
    pat_seqs
        List of sequences to use as xticklabels.
    dendrogram_ratio
        Ratio of dendograms in x and y directions.
    importance_threshold
        Minimal pattern importance threshold over all classes to retain the pattern before clustering and plotting.

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
        figsize=figsize,
        row_colors=None,
        yticklabels=classes,
        center=center,
        xticklabels=True
        if pat_seqs is None
        else False,  # Disable default xticklabels if pat_seqs provided.  #xticklabels=xtick_labels,
        method=method,
        dendrogram_ratio=dendrogram_ratio,
        cbar_pos=(1.05, 0.4, 0.01, 0.3),
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

        constant = (1 / figsize[1]) * 64
        for i, (letters, scores) in enumerate(reordered_pat_seqs):
            previous_spacing = 0
            for _, (letter, score) in enumerate(
                zip(reversed(letters), reversed(scores))
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

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')

    plt.show()


def selected_instances(pattern_dict: dict, idcs: list[int]) -> None:
    """
    Plot the patterns specified by the indices in `idcs` from the `pattern_dict`.

    Parameters
    ----------
    pattern_dict
        A dictionary containing pattern data. Each key corresponds to a pattern ID, and the value is a nested structure containing
        contribution scores and metadata for the pattern. Refer to the output of `crested.tl.modisco.process_patterns`.
    idcs
        A list of indices specifying which patterns to plot. The indices correspond to keys in the `pattern_dict`.

    See Also
    --------
    crested.tl.modisco.process_patterns

    Examples
    --------
    >>> pattern_indices = [0, 1, 2]
    >>> crested.pl.patterns.selected_instances(pattern_dict, pattern_indices)

    .. image:: ../../../../docs/_static/img/examples/pattern_selected_instances.png
    """
    figure, axes = plt.subplots(nrows=len(idcs), ncols=1, figsize=(8, 2 * len(idcs)))
    if len(idcs) == 1:
        axes = [axes]

    for i, idx in enumerate(idcs):
        ax = _plot_attribution_map(
            ax=axes[i],
            saliency_df=np.array(pattern_dict[str(idx)]["pattern"]["contrib_scores"]),
            return_ax=True,
            figsize=None,
        )
        ax.set_title(pattern_dict[str(idx)]["pattern"]["id"])

    plt.tight_layout()
    plt.show()


def class_instances(
    pattern_dict: dict, idx: int, class_representative: bool = False
) -> None:
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
    figure, axes = plt.subplots(
        nrows=n_instances, ncols=1, figsize=(8, 2 * n_instances)
    )
    if n_instances == 1:
        axes = [axes]

    instance_classes = list(pattern_dict[str(idx)][key].keys())

    for i, cl in enumerate(instance_classes):
        ax = _plot_attribution_map(
            ax=axes[i],
            saliency_df=np.array(pattern_dict[str(idx)][key][cl]["contrib_scores"]),
            return_ax=True,
            figsize=None,
        )
        ax.set_title(pattern_dict[str(idx)][key][cl]["id"])

    plt.tight_layout()
    plt.show()


def similarity_heatmap(
    similarity_matrix: np.ndarray,
    indices: list,
    fig_size: tuple[int, int] = (30, 15),
    fig_path: str | None = None,
) -> None:
    """
    Plot a similarity heatmap of all pattern indices.

    Parameters
    ----------
    similarity_matrix
        A 2D numpy array containing the similarity values.
    indices
        List of pattern indices.
    fig_size
        Size of the figure for the heatmap.
    fig_path
        Path to save the figure. If None, the figure will be shown but not saved.

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
    fig, ax = plt.subplots(figsize=fig_size)
    heatmap = sns.heatmap(
        similarity_matrix,
        ax=ax,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        xticklabels=indices,
        yticklabels=indices,
        annot_kws={"size": 8},
    )

    for _, spine in heatmap.spines.items():
        spine.set_visible(True)
        spine.set_color("grey")
        spine.set_linewidth(0.5)

    plt.title("Pattern Similarity Heatmap", fontsize=20)
    plt.xlabel("Pattern Index", fontsize=15)
    plt.ylabel("Pattern Index", fontsize=15)

    if fig_path is not None:
        plt.savefig(fig_path)
    plt.show()


def tf_expression_per_cell_type(
    df: pd.DataFrame,
    tf_list: list,
    log_transform: bool = False,
    title: str = "TF Expression per Cell Type",
) -> None:
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
    title
        The title of the plot.
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
        tf_expression_df = np.log(tf_expression_df + 1)

    # Plot the TF expression per cell type
    ax = tf_expression_df.plot(kind="bar", figsize=(12, 5), width=0.8)
    ax.set_title(title)
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Log Mean TF Expression" if log_transform else "Mean TF Expression")
    ax.legend(title="Transcription Factors")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def clustermap_tf_motif(
    data: np.ndarray,
    heatmap_dim: str = "gex",
    dot_dim: str = "contrib",
    class_labels: list[str] | None = None,
    subset_classes: list[str] | None = None,
    pattern_labels: list[str] | None = None,
    fig_size: tuple[int, int] | None = None,
    save_path: str | None = None,
    cluster_rows: bool = True,
    cluster_columns: bool = True,
) -> None:
    """
    Generate a heatmap where one modality is represented as color, and the other as dot size.

    Parameters
    ----------
    data : numpy.ndarray
        3D numpy array with shape (len(classes), #patterns, 2).
    heatmap_dim : str
        Either 'gex' or 'contrib', indicating which third dimension to use for heatmap colors.
    dot_dim : str
        Either 'gex' or 'contrib', indicating which third dimension to use for dot sizes.
    class_labels : list[str] | None
        Labels for the classes.
    subset_classes : list[str] | None
        Subset of classes to include in the heatmap. Rows in `data` are filtered accordingly.
    pattern_labels : list[str] | None
        Labels for the patterns.
    fig_size : tuple[int, int] | None
        Size of figure. If None, it will be auto-configured.
    save_path : str | None
        File path to save figure to.
    cluster_rows : bool
        Whether to cluster the rows (classes). Default is True.
    cluster_columns : bool
        Whether to cluster the columns (patterns). Default is True.

    Examples
    --------
    >>> clustermap_tf_motif_v2(
    ...     data,
    ...     heatmap_dim="gex",
    ...     dot_dim="contrib",
    ...     class_labels=classes,
    ...     pattern_labels=patterns,
    ...     cluster_rows=True,
    ...     cluster_columns=True,
    ... )
    """
    assert data.shape[2] == 2, "The third dimension of the data must be 2."

    # Set default labels if not provided
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(data.shape[0])]
    if pattern_labels is None:
        pattern_labels = [f"Pattern {i}" for i in range(data.shape[1])]

    # Subset classes if specified
    if subset_classes is not None:
        subset_indices = [i for i, label in enumerate(class_labels) if label in subset_classes]
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
    pattern_labels = [label for i, label in enumerate(pattern_labels) if valid_columns[i]]

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
    if fig_size is None:
        fig_size = (max(20, data.shape[1] // 4), data.shape[0] // 2)
    fig = plt.figure(figsize=fig_size)

    if cluster_rows:
        gs = fig.add_gridspec(1, 2, width_ratios=[0.2, 4], wspace=0.075)
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
        vmax=max(np.abs(heatmap_data.min()), np.abs(heatmap_data.max()))
    )

    # Plot heatmap
    heatmap = ax_heatmap.imshow(
        heatmap_data,
        aspect="auto",
        cmap="coolwarm",
        norm=norm,
    )

    # Overlay dots
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax_heatmap.scatter(
                j, i, s=dot_size_data[i, j] * 100, c="black", alpha=0.6
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

    # Final layout adjustments
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
