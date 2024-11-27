"""Fixtures to be used by all unit tests."""

import os

import genomepy
import keras
import pytest

from ._utils import create_anndata_with_regions


@pytest.fixture(scope="module")
def keras_model():
    """Keras model fixture."""
    from crested.tl.zoo import simple_convnet

    model = simple_convnet(
        seq_len=500,
        num_classes=5,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    return model


@pytest.fixture(scope="module")
def adata():
    """Anndata fixture."""
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
    """Genome fixture."""
    if not os.path.exists("tests/data/genomes/hg38.fa"):
        genomepy.install_genome(
            "hg38", annotation=False, provider="UCSC", genomes_dir="tests/data/genomes"
        )
    return "tests/data/genomes/hg38/hg38.fa"
