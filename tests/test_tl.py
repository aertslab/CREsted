"""Test tl module."""

import os

import genomepy
import keras
import numpy as np
import pytest

import crested
from crested.tl._tools import _transform_input, detect_input_type

from ._utils import create_anndata_with_regions


@pytest.fixture(scope="module")
def keras_model():
    from crested.tl.zoo import simple_convnet

    model = simple_convnet(
        seq_len=500,
        num_classes=10,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    return model


@pytest.fixture(scope="module")
def adata():
    regions = [
        "chr1:194208032-194208532",
        "chr1:92202766-92203266",
        "chr1:92298990-92299490",
        "chr1:3406052-3406552",
        "chr1:183669567-183670067",
        "chr1:109912183-109912683",
        "chr1:92210697-92211197",
        "chr1:59100954-59101454",
        "chr1:84634055-84634555",
        "chr1:48792527-48793027",
    ]
    return create_anndata_with_regions(regions)


@pytest.fixture(scope="module")
def genome():
    if not os.path.exists("tests/data/genomes/hg38.fa"):
        genomepy.install_genome(
            "hg38", annotation=False, provider="UCSC", genomes_dir="tests/data/genomes"
        )
    return "tests/data/genomes/hg38/hg38.fa"


def test_input_type(adata):
    assert detect_input_type(adata) == "anndata"
    assert detect_input_type("chr1:1-100") == "region"
    assert detect_input_type("ACGT") == "sequence"
    assert detect_input_type(["chr1:1-100", "chr1:101-200"]) == "region"
    assert detect_input_type(["ACGT", "ACGT"]) == "sequence"
    with pytest.raises(ValueError):
        detect_input_type(["chr1:1-100", "ACGT"])
    with pytest.raises(ValueError):
        detect_input_type(["chr1:1-100", 1])
    with pytest.raises(ValueError):
        detect_input_type([1, 2])
    with pytest.raises(ValueError):
        detect_input_type([1, "ACGT"])
    with pytest.raises(ValueError):
        detect_input_type(1)
    with pytest.raises(ValueError):
        detect_input_type(1.0)
    with pytest.raises(ValueError):
        detect_input_type(None)


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
        input, keras_model, genome=genome, layer_name="denseblock_dense"
    )
    assert embeddings.shape == (2, 8)
