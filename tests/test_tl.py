"""Test tl module."""

import numpy as np
import pytest

import crested
from crested.tl._tools import _detect_input_type, _transform_input


def test_input_type(adata):
    assert _detect_input_type(adata) == "anndata"
    assert _detect_input_type("chr1:1-100") == "region"
    assert _detect_input_type("ACGT") == "sequence"
    assert _detect_input_type(["chr1:1-100", "chr1:101-200"]) == "region"
    assert _detect_input_type(["ACGT", "ACGT"]) == "sequence"
    with pytest.raises(ValueError):
        _detect_input_type(["chr1:1-100", "ACGT"])
    with pytest.raises(ValueError):
        _detect_input_type(["chr1:1-100", 1])
    with pytest.raises(ValueError):
        _detect_input_type([1, 2])
    with pytest.raises(ValueError):
        _detect_input_type([1, "ACGT"])
    with pytest.raises(ValueError):
        _detect_input_type(1)
    with pytest.raises(ValueError):
        _detect_input_type(1.0)
    with pytest.raises(ValueError):
        _detect_input_type(None)


def test_input_transform(genome):
    assert _transform_input("AACGT").shape == (1, 5, 4)
    assert np.array_equal(
        _transform_input("ACGT"),
        np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]),
    )
    assert np.array_equal(
        _transform_input(["ACGT", "ACGT"]),
        np.array(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ]
        ),
    )
    assert np.array_equal(
        _transform_input("chr1:1-6", genome), np.array([[[0, 0, 0, 0]] * 5])
    )
    assert _transform_input(np.array([[[1, 0, 0, 0]]])).shape == (1, 1, 4)


def test_get_embeddings(keras_model, genome):
    input = "ATCGA" * 100
    embeddings = crested.tl.get_embeddings(
        input, keras_model, genome=genome, layer_name="denseblock_dense"
    )
    assert embeddings.shape == (1, 8)
    input = ["ATCGA" * 100, "ATCGA" * 100]
    embeddings = crested.tl.get_embeddings(
        input, keras_model, layer_name="denseblock_dense"
    )
    assert embeddings.shape == (2, 8)


def test_predict(keras_model, adata, genome):
    input = "ATCGA" * 100
    predictions = crested.tl.predict(input, keras_model)
    assert predictions.shape == (1, 5)

    predictions = crested.tl.predict(input=adata, model=keras_model, genome=genome)
    assert predictions.shape == (10, 5)

    models = [keras_model, keras_model]
    predictions = crested.tl.predict(input=adata, model=models, genome=genome)
    assert predictions.shape == (10, 5)


def test_score_gene_locus(keras_model, adata, genome):
    gene_locus = "chr1:200000-200500"
    scores, coordinates, min_loc, max_loc, tss_pos = crested.tl.score_gene_locus(
        gene_locus=gene_locus,
        all_class_names=list(adata.obs_names),
        class_name=list(adata.obs_names)[0],
        model=keras_model,
        genome=genome,
        upstream=1000,
        downstream=1000,
        step_size=500,
    )
    assert scores.shape == (2500,), scores.shape
    assert coordinates.shape == (int(2500 / 500), 3), coordinates.shape
    assert min_loc == 199000
    assert max_loc == 201500
    assert tss_pos == 200000
