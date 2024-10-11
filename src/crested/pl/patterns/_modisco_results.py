from __future__ import annotations

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.cluster.hierarchy import leaves_list, linkage

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

    # Remove columns that contain only zero values
    non_zero_columns = np.any(pattern_matrix != 0, axis=0)
    pattern_matrix = pattern_matrix[:, non_zero_columns]

    # Reindex columns based on the original positions of non-zero columns
    column_indices = np.where(non_zero_columns)[0]
    data = pd.DataFrame(pattern_matrix, columns=column_indices)

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
        dendrogram_ratio=(0.1, 0.1),
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
        reordered_pat_seqs = [pat_seqs[column_indices[i]] for i in col_order]
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
        plt.savefig(fig_path)

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
    cluster_on_dim: str = "gex",
    class_labels: None | list[str] = None,
    pattern_labels: None | list[str] = None,
    color_idx: str = "gex",
    size_idx: str = "contrib",
    grid: bool = True,
    log_transform: bool = False,
    normalize: bool = False,
) -> None:
    """
    Plot a clustermap from a 3D matrix where one third dimension is indicated by dot sizen and the other by color.

    Parameters
    ----------
    data
        3D numpy array with shape (len(classes), #patterns, 2)
    cluster_on_dim
        either 'gex' or 'contrib', indicating which third dimension to cluster on
    class_labels
        labels for the classes
    pattern_labels
        labels for the patterns
    color_idx
        either 'gex' or 'contrib', indicating the dimension to use for color
    size_idx
        either 'gex' or 'contrib', indicating the dimension to use for size
    grid
        whether to add a grid to the figure
    log_transform
        whether to apply log transformation to the data
    normalize
         whether to normalize the data

    See Also
    --------
    crested.tl.modisco.create_tf_ct_matrix

    Examples
    --------
    >>> crested.pl.patterns.clustermap_tf_motif(
    ...     tf_ct_matrix,
    ...     cluster_on_dim="gex",
    ...     class_labels=classes,
    ...     pattern_labels=tf_pattern_annots,
    ...     color_idx="gex",
    ...     size_idx="contrib",
    ... )

    .. image:: ../../../../docs/_static/img/examples/pattern_tf_motif_clustermap.png
    """
    # Ensure data is a numpy array
    data = np.array(data)
    assert data.shape[2] == 2, "The third dimension of the data should be 2."

    # Some additional data prep for more logical plotting
    if color_idx == "gex":
        for col_idx in range(data.shape[1]):
            for ct_idx in range(data.shape[0]):
                if data[ct_idx, col_idx, 1] < 0:
                    data[ct_idx, col_idx, 0] = -data[ct_idx, col_idx, 0]
                    data[ct_idx, col_idx, 1] = np.abs(data[ct_idx, col_idx, 1])

    # Default labels if none provided
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(data.shape[0])]
    if pattern_labels is None:
        pattern_labels = [f"Pattern {i}" for i in range(data.shape[1])]

    # Mapping from string to index
    dim_mapping = {"gex": 0, "contrib": 1}

    # Choose the dimension to cluster on
    clustering_data = data[:, :, dim_mapping[cluster_on_dim]]

    if log_transform:
        clustering_data = np.log(clustering_data)

    if normalize:
        clustering_data = clustering_data / np.linalg.norm(
            clustering_data, axis=1, keepdims=True
        )

    # Perform hierarchical clustering
    linkage_matrix = linkage(clustering_data, method="ward")
    cluster_order = leaves_list(linkage_matrix)

    # Reorder data according to clustering
    data_ordered = data[cluster_order, :, :]

    # Extract the two dimensions
    size_data = data_ordered[:, :, dim_mapping[size_idx]]
    max_size = np.max(size_data)
    size_data = size_data / max_size
    color_data = data_ordered[:, :, dim_mapping[color_idx]]

    # Adjust figure size dynamically
    fig, ax = plt.subplots(figsize=(max(20, data.shape[1] // 4), 10))

    # Determine color scale limits to center on zero
    max_val = np.max(color_data)
    min_val = np.min(color_data)

    # Define the normalization to center at zero
    norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)

    # Use the norm parameter in scatter
    sc = ax.scatter(
        np.tile(np.arange(data_ordered.shape[1]), data_ordered.shape[0]),
        np.repeat(np.arange(data_ordered.shape[0]), data_ordered.shape[1]),
        s=size_data.flatten() * 500,
        c=color_data.flatten(),
        cmap="coolwarm",
        alpha=0.6,
        norm=norm,  # Apply the centered colormap
    )

    # Add color bar
    cbar = plt.colorbar(sc, ax=ax)
    label = (
        "Average pattern contribution score"
        if dim_mapping[color_idx] == 1
        else "Average TF expression, signed by activation/repression"
    )
    cbar.set_label(label)

    # Set labels
    ax.set_xticks(np.arange(data_ordered.shape[1]))
    ax.set_yticks(np.arange(data_ordered.shape[0]))

    # Reduce the number of x-axis labels displayed
    ax.set_xticklabels(
        [pattern_labels[i] for i in range(data_ordered.shape[1])], rotation=90
    )
    ax.set_yticklabels([class_labels[i] for i in cluster_order])
    ax.set_xlim([-0.5, len(pattern_labels) + 0.5])

    plt.xlabel("Patterns")
    plt.ylabel("Classes")
    plt.grid(grid)
    plt.tight_layout()
    plt.show()
