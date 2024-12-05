"""Fixtures to be used by all unit tests."""

import os

import genomepy
import keras
import numpy as np
import pytest

np.random.seed(42)
keras.utils.set_random_seed(42)


@pytest.fixture(scope="module")
def genome():
    """Genome fixture."""
    if not os.path.exists("tests/data/genomes/hg38.fa"):
        genomepy.install_genome(
            "hg38", annotation=False, provider="UCSC", genomes_dir="tests/data/genomes"
        )
    return "tests/data/genomes/hg38/hg38.fa"
