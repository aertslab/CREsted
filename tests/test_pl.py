import inspect

import numpy as np
import pytest

import crested.pl


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


if __name__ == "__main__":
    pytest.main()
