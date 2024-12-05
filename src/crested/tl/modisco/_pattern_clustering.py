from __future__ import annotations
import numpy as np
import igraph as ig
import leidenalg as la
from loguru import logger
from ._modisco_utils import (
    read_modisco_results,
    get_ic_modisco,
    trim_by_ic,
    filter_patterns,
    convert_seqletSet_to_dict,
)
from ._tangermeme_utils import add_padding_to_matrix, run_pairwise_tomtom


def construct_pattern_graph(
    evalues: np.ndarray,
    length_patterns: list,
    overlaps: np.ndarray,
    max_evalue: float = 1e-2,
    pseudocount: float = 1e-8,
    min_overlap: float = 0.6,
) -> ig.Graph:
    """
    Create a graph based on tomtom e-values

    Parameters
    ----------
    evalues:
    matrix containing tomtom e-values

    lengh_patterns:
    list containing the lengths of trimmed pattern (before padding with 0s)

    max_evalue:
    max. allowed e-value for adding edge

    pseuodcount:
    to be added to avoid taking log2 of 0 when converting e-values into edge weights

    min_overlap:
    min. fraction of overlapping bases against trimmed pattern size
    """
    edges = []
    weights = []

    n_patterns = len(length_patterns)

    for idx1 in range(n_patterns - 1):
        for idx2 in range(idx1 + 1, n_patterns):
            evalue = evalues[idx1][idx2]

            # chek motif sizes and overlap
            len_pat1 = length_patterns[idx1]
            len_pat2 = length_patterns[idx2]
            overlap = overlaps[idx1][idx2]
            ratio_overlap = overlap / np.max([len_pat1, len_pat2])

            if ratio_overlap < min_overlap:
                continue

            if evalue < max_evalue:
                edge = (idx1, idx2)
                weight = -np.log2(evalue + pseudocount)

                if weight > 0:
                    edges.append(edge)
                    weights.append(weight)

    g = ig.Graph()
    g.add_vertices(n_patterns)
    g.add_edges(edges)
    g.es["weight"] = weights

    return g


def leiden_cluster_graph(
    graph: ig.Graph,
    resolution: float = 1,
    n_iterations: int = 10,
    seed: int = 42,
):
    """
    Perform Leiden clustering on graph

    Parameters
    ----------
    graph:
    graph for which partition should be dected

    resolution:
    resolution parameter for Leiden clustering

    seed: random seed for RNG
    """
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        seed=seed,
        n_iterations=n_iterations,
        resolution_parameter=resolution,
    )

    return partition


def get_best_pagerank_per_group(
    graph: ig.Graph,
    idx_groups: list[list],
    is_directed: bool = False,
) -> (list, dict):
    list_best_idx = []
    pageranks = {}

    for group_idxs in idx_groups:
        best_idx = group_idxs[0]
        if len(group_idxs) > 1:
            temp_pageranks = graph.pagerank(
                vertices=group_idxs, weights=graph.es["weight"], directed=is_directed
            )
            max_pagerank = 0
            for i, idx in enumerate(group_idxs):
                pageranks[idx] = temp_pageranks[i]
                if pageranks[idx] > max_pagerank:
                    max_pagerank = pageranks[idx]
                    best_idx = idx
        else:
            pageranks[group_idxs[0]] = 0
        list_best_idx.append(best_idx)

    return list_best_idx, pageranks


def leiden_cluster_patterns(
    evalues: np.ndarray,
    overlaps: np.ndarray,
    pattern_lengths: list,
    max_evalue: float = 1e-2,
    pseudocount: float = 1e-8,
    min_overlap: float = 0.6,
    resolution: float = 1,
) -> (list[list[int]], list[str], list[int], list[float]):
    pattern_graph = construct_pattern_graph(
        evalues,
        pattern_lengths,
        overlaps,
        max_evalue=max_evalue,
        min_overlap=min_overlap,
    )
    partition = leiden_cluster_graph(pattern_graph, resolution)

    list_cluster_member = [str(clus) for clus in list(partition.membership)]
    dict_cluster_members = {}
    for idx_p, clus in enumerate(list_cluster_member):
        if clus in dict_cluster_members.keys():
            dict_cluster_members[clus].append(idx_p)
        else:
            dict_cluster_members[clus] = [idx_p]
    name_clusters = list(dict_cluster_members.keys())
    list_clusters = list(dict_cluster_members.values())
    best_patterns, pageranks = get_best_pagerank_per_group(pattern_graph, list_clusters)

    return list_clusters, name_clusters, best_patterns, pageranks


def create_pattern_cluster_dict(
    clusters_idx: list[list[int]],
    best_pattern_idx: list[int],
    pattern_names: list[str],
    names_clusters: list[str],
    seqletSets: dict,
    averageIC: dict,
    pageranks: dict | None = None,
    separator="___",
    no_singlets: bool = False,
) -> dict:
    pattern_clusters = {}
    # loop over clusters
    for idx, clus in enumerate(names_clusters):
        classes = []
        nseqlets_class = {}

        name = clus
        if len(clusters_idx[idx]) == 1:
            if no_singlets:
                continue
            else:
                #   name = "singlet" + separator + clus
                pattern_clusters[name] = {}
                pattern_clusters[name]["is_singlet"] = True
        else:
            pattern_clusters[name] = {}
            pattern_clusters[name]["is_singlet"] = False

        # representative pattern entry
        pattern_clusters[name]["pattern"] = convert_seqletSet_to_dict(
            seqletSets[pattern_names[best_pattern_idx[idx]]]
        )
        pattern_clusters[name]["pattern"]["id"] = pattern_names[best_pattern_idx[idx]]
        mclass, activity, bestpat = pattern_names[best_pattern_idx[idx]].split(
            separator
        )

        if activity == "(+)":
            pattern_clusters[name]["pattern"]["pos_pattern"] = True
        else:
            pattern_clusters[name]["pattern"]["pos_pattern"] = False

        pattern_clusters[name]["instances"] = {
            pattern_names[cidx]: convert_seqletSet_to_dict(
                seqletSets[pattern_names[cidx]]
            )
            for cidx in clusters_idx[idx]
        }
        pattern_clusters[name]["pattern"]["ppm"] = pattern_clusters[name]["pattern"][
            "sequence"
        ].copy()

        if averageIC is not None:
            pattern_clusters[name]["pattern"]["ic"] = averageIC[
                pattern_names[best_pattern_idx[idx]]
            ]
        if pageranks is not None:
            pattern_clusters[name]["pattern"]["pagerank"] = pageranks[
                pattern_names[best_pattern_idx[idx]]
            ]

        # update cluster memmber entries
        for pat in pattern_clusters[name]["instances"].keys():
            mclass, activity, patname = pat.split(separator)
            pattern_clusters[name]["instances"][pat]["id"] = pat
            classes.append(mclass)

            if mclass in nseqlets_class.keys():
                nseqlets_class[mclass] += len(
                    pattern_clusters[name]["instances"][pat]["seqlets"]
                )
            else:
                nseqlets_class[mclass] = len(
                    pattern_clusters[name]["instances"][pat]["seqlets"]
                )

            pattern_clusters[name]["instances"][pat]["class"] = mclass
            if activity == "(+)":
                pattern_clusters[name]["instances"][pat]["pos_pattern"] = True
            else:
                pattern_clusters[name]["instances"][pat]["pos_pattern"] = False
            if averageIC is not None:
                pattern_clusters[name]["instances"][pat]["ic"] = averageIC[pat]
            if pageranks is not None:
                pattern_clusters[name]["instances"][pat]["pagerank"] = pageranks[pat]

        # add seqlet count per class
        pattern_clusters[name]["classes"] = {}
        for cl, count in nseqlets_class.items():
            pattern_clusters[name]["classes"][cl] = {}
            pattern_clusters[name]["classes"][cl]["n_seqlets"] = count

        # select representative class pattern
        for cl in pattern_clusters[name]["classes"].keys():
            class_patterns = [
                pat
                for pat in pattern_clusters[name]["instances"].keys()
                if pattern_clusters[name]["instances"][pat]["class"] == cl
            ]

            if pageranks is not None:
                class_prs = [pageranks[pat] for pat in class_patterns]
                selected_pat = class_patterns[np.argmax(class_prs)]
            elif averageIC is not None:
                class_ics = [averageIC[pat] for pat in class_patterns]
                selected_pat = class_patterns[np.argmax(class_ics)]

            for key in pattern_clusters[name]["instances"][selected_pat].keys():
                if type(pattern_clusters[name]["instances"][selected_pat][key]) not in [
                    str,
                    int,
                    float,
                    bool,
                ]:
                    pattern_clusters[name]["classes"][cl][key] = pattern_clusters[name][
                        "instances"
                    ][selected_pat][key].copy()
                else:
                    pattern_clusters[name]["classes"][cl][key] = pattern_clusters[name][
                        "instances"
                    ][selected_pat][key]
    total_clusters = len(pattern_clusters.keys())
    pattern_clusters_reindexed = {}
    for i in range(total_clusters):
        pattern_clusters_reindexed[str(i)] = pattern_clusters[list(pattern_clusters.keys())[i]]
    pattern_clusters=pattern_clusters_reindexed

    return pattern_clusters


def process_patterns_leiden(
    matched_files: dict[str, str | list[str] | None],
    trim_ic_threshold: float = 0.25,
    discard_ic_threshold: float = 1,
    min_size=5,
    pattern_size=30,
    max_evalue: float = 0.01,
    min_overlap: float = 0,
    resolution: float = 1,
    separator: str = "___",
    verbose: bool = False,
) -> dict:
    """
    Process genomic patterns from matched HDF5 files, trim based on information content, and match to known patterns.

    Parameters
    ----------
    matched_files
        dictionary with class names as keys and paths to HDF5 files as values.
    sim_threshold
        Similarity threshold for matching patterns (-log10(pval), pval obtained through TOMTOM matching from tangermeme)
    trim_ic_threshold
        Information content threshold for trimming patterns.
    discard_ic_threshold
        Information content threshold for discarding patterns.
    verbose
        Flag to enable verbose output.

    See Also
    --------
    crested.tl.modisco.match_h5_files_to_classes

    Returns
    -------
    All processed patterns with metadata.
    """
    all_patterns = {}
    trimmed_patterns = {}
    filtered_patterns = {}
    pattern_clusters = {}

    # read patterns
    for cell_type in matched_files:
        if matched_files[cell_type] is None:
            continue

        pos_patterns, neg_patterns = read_modisco_results(matched_files[cell_type])

        for pattern in pos_patterns.keys():
            patname = (
                cell_type.replace(" ", "_") + separator + "(+)" + separator + pattern
            )
            all_patterns[patname] = pos_patterns[pattern].copy()
        for pattern in neg_patterns.keys():
            patname = (
                cell_type.replace(" ", "_") + separator + "(-)" + separator + pattern
            )
            all_patterns[patname] = neg_patterns[pattern].copy()

    # trim patterns
    for pattern in all_patterns:
        trimmed_patterns[pattern] = trim_by_ic(
            all_patterns[pattern], ic_threshold=trim_ic_threshold
        )

    # compute average IC
    dict_avgIC = {
        pattern: np.mean(get_ic_modisco(seqlet))
        for pattern, seqlet in trimmed_patterns.items()
    }

    # filter patterns
    filtered_patterns = filter_patterns(
        trimmed_patterns, thr_avg_ic=discard_ic_threshold, min_size=min_size
    )

    # pairwise pattern comparison with tangermeme
    patterns_name = list(filtered_patterns.keys())
    patterns_length = [pat.sequence.shape[0] for pat in filtered_patterns.values()]
    patterns_pwm = [
        add_padding_to_matrix(pat.sequence, desired_size=pattern_size)
        for pat in filtered_patterns.values()
    ]
    pvals, scores, offsets, overlaps, strands, evals = run_pairwise_tomtom(patterns_pwm)

    # cluster patterns
    list_clusters, name_clusters, best_patterns, pageranks = leiden_cluster_patterns(
        evals,
        overlaps,
        patterns_length,
        max_evalue=max_evalue,
        min_overlap=min_overlap,
        resolution=resolution,
    )

    # construct dictionary
    pattern_clusters = create_pattern_cluster_dict(
        list_clusters,
        best_patterns,
        patterns_name,
        name_clusters,
        filtered_patterns,
        dict_avgIC,
    )

    return pattern_clusters
