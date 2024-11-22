import inspect
import os
import shutil

import numpy as np
import pytest

import crested.pl

from ._utils import generate_simulated_patterns


def test_plot_functions_use_render_plot():
    # Get all submodules in crested.pl
    submodules = [crested.pl]
    for _, obj in inspect.getmembers(crested.pl):
        if inspect.ismodule(obj):
            submodules.append(obj)

    # Check each function in the submodules
    for submodule in submodules:
        for name, func in inspect.getmembers(submodule, inspect.isfunction):
            # Skip private functions and render_plot function itself
            if name.startswith("_") or name == "render_plot":
                continue

            # Get the source code of the function
            source = inspect.getsource(func)

            # Check if render_plot is used in the function
            assert (
                "render_plot" in source
            ), f"Function {name} in {submodule.__name__} does not use render_plot"


def test_locus_scoring_without_bigwig():
    scores = np.random.rand(100)
    range_values = (0, 100)
    gene_start = 20
    gene_end = 40

    fig = crested.pl.hist.locus_scoring(
        scores=scores,
        range=range_values,
        gene_start=gene_start,
        gene_end=gene_end,
        bigwig_values=None,
        bigwig_midpoints=None,
        show=False,
    )
    assert fig is not None


def test_locus_scoring_with_bigwig():
    scores = np.random.rand(100)
    range_values = (0, 100)
    gene_start = 20
    gene_end = 40
    bigwig_values = np.random.rand(50)
    bigwig_midpoints = np.linspace(0, 100, 50)

    fig = crested.pl.hist.locus_scoring(
        scores=scores,
        range=range_values,
        gene_start=gene_start,
        gene_end=gene_end,
        bigwig_values=bigwig_values,
        bigwig_midpoints=bigwig_midpoints,
        show=False,
    )
    assert fig is not None


@pytest.fixture(scope="module")
def all_patterns():
    return generate_simulated_patterns()


@pytest.fixture(scope="module")
def all_classes():
    return [
        "Astro",
        "Endo",
        "L2_3IT",
        "L5ET",
        "L5IT",
        "L5_6NP",
        "L6CT",
        "L6IT",
        "L6b",
        "Micro_PVM",
        "Oligo",
        "Pvalb",
        "Sst",
        "SstChodl",
        "VLMC",
        "Lamp5",
        "OPC",
        "Sncg",
        "Vip",
    ]


@pytest.fixture(scope="module")
def pattern_matrix(all_patterns, all_classes):
    pattern_matrix = crested.tl.modisco.create_pattern_matrix(
        classes=all_classes, all_patterns=all_patterns, normalize=True
    )
    return pattern_matrix


@pytest.fixture(scope="module")
def save_dir():
    path = "tests/data/pl_output"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def test_patterns_clustermap(all_patterns, all_classes, pattern_matrix, save_dir):
    pat_seqs = crested.tl.modisco.generate_nucleotide_sequences(all_patterns)
    save_path = os.path.join(save_dir, "patterns_clustermap.png")
    crested.pl.patterns.clustermap(
        pattern_matrix,
        classes=all_classes,
        subset=["Astro", "OPC", "Oligo"],
        pat_seqs=pat_seqs,
        grid=True,
        save_path=save_path,
        height=2,
        width=20,
    )


def test_patterns_selected_instances(all_patterns, save_dir):
    pattern_indices = [0, 1]
    save_path = os.path.join(save_dir, "patterns_selected_instances.png")
    crested.pl.patterns.selected_instances(
        all_patterns, pattern_indices, save_path=save_path
    )


if __name__ == "__main__":
    pytest.main()
