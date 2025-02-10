"""Test that ensures the outputs after the functional refactor are the same as before."""

import os

import keras
import numpy as np
import pytest

from crested.tl import (
    Crested,
    contribution_scores,
    contribution_scores_specific,
    extract_layer_embeddings,
    predict,
    score_gene_locus,
)
from crested.tl.data import AnnDataModule

np.random.seed(42)
keras.utils.set_random_seed(42)

if os.environ["KERAS_BACKEND"] == "tensorflow":
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()


@pytest.fixture(scope="module")
def crested_object(keras_model, adata, genome):
    anndatamodule = AnnDataModule(
        adata,
        genome=genome,
        batch_size=32,
        always_reverse_complement=False,
        deterministic_shift=False,
        shuffle=False,
    )
    crested_object = Crested(
        data=anndatamodule,
    )
    crested_object.model = keras_model
    return crested_object


@pytest.fixture(scope="module")
def crested_object_specific(keras_model, adata_specific, genome):
    anndatamodule = AnnDataModule(
        adata_specific,
        genome=genome,
        batch_size=32,
        always_reverse_complement=False,
        deterministic_shift=False,
        shuffle=False,
    )
    crested_object = Crested(
        data=anndatamodule,
    )
    crested_object.model = keras_model
    return crested_object


def test_predict_adata(adata, crested_object, keras_model, genome):
    crested_object_preds = crested_object.predict()
    refactored_preds = predict(adata, keras_model, genome)
    assert np.allclose(
        crested_object_preds,
        refactored_preds,
        atol=1e-4,
    ), "Anndata predictions are not equal."


def test_predict_sequence(crested_object, keras_model):
    sequence = "ATCGA" * 100
    crested_object_preds = crested_object.predict_sequence(sequence)
    refactored_preds = predict(sequence, keras_model)
    assert np.allclose(
        crested_object_preds,
        refactored_preds,
        atol=1e-4,
    ), "Sequence predictions are not equal"


def test_predict_regions(crested_object, keras_model, genome):
    regions = ["chr1:1-501", "chr1:101-601"]
    crested_object_preds = crested_object.predict_regions(region_idx=regions)
    refactored_preds = predict(regions, keras_model, genome)
    assert np.allclose(
        crested_object_preds,
        refactored_preds,
        atol=1e-4,
    ), "Region predictions are not equal"


def test_extract_layer_embeddings(adata, crested_object, keras_model, genome):
    crested_object_embeddings = crested_object.get_embeddings(
        layer_name="denseblock_dense"
    )
    refactored_embeddings = extract_layer_embeddings(
        input=adata,
        model=keras_model,
        genome=genome,
        layer_name="denseblock_dense",
    )
    assert np.allclose(
        crested_object_embeddings,
        refactored_embeddings,
        atol=1e-4,
    ), "Embeddings are not equal."


def test_score_gene_locus(crested_object, adata, keras_model, genome):
    chrom_name = "chr1"
    gene_start = "2000000"
    gene_end = "2002000"
    scores, coordinates, min_loc, max_loc, tss_pos = crested_object.score_gene_locus(
        chr_name=chrom_name,
        gene_start=int(gene_start),
        gene_end=int(gene_end),
        class_name=list(adata.obs_names)[0],
        window_size=500,
        downstream=2000,
        upstream=2000,
    )
    (
        ref_scores,
        ref_coordinates,
        ref_min_loc,
        ref_max_loc,
        ref_tss_pos,
    ) = score_gene_locus(
        chr_name=chrom_name,
        gene_start=int(gene_start),
        gene_end=int(gene_end),
        target_idx=0,
        model=keras_model,
        genome=genome,
        downstream=2000,
        upstream=2000,
    )
    assert np.allclose(
        scores,
        ref_scores,
        atol=1e-4,
    ), "Scores are not equal."
    assert (
        a == b for a, b in zip(coordinates, ref_coordinates)
    ), "Coordinates are not equal."
    assert min_loc == ref_min_loc, "Minimum location is not equal."
    assert max_loc == ref_max_loc, "Maximum location is not equal."
    assert tss_pos == ref_tss_pos, "TSS position is not equal."


def test_contribution_scores_region(crested_object, adata, keras_model, genome):
    region = ["chr1:1-501", "chr1:101-601"]
    (
        scores,
        one_hot_encoded_sequences,
    ) = crested_object.calculate_contribution_scores_regions(
        region_idx=region,
        class_names=list(adata.obs_names)[0:2],
        method="integrated_grad",
    )
    scores_refactored, one_hot_encoded_sequences_refactored = contribution_scores(
        input=region,
        target_idx=[0, 1],
        model=keras_model,
        genome=genome,
        method="integrated_grad",
    )
    assert np.allclose(
        scores,
        scores_refactored,
        atol=1e-5,
    ), "Scores are not equal."
    assert np.array_equal(
        one_hot_encoded_sequences,
        one_hot_encoded_sequences_refactored,
    ), "One-hot encoded sequences are not equal"


def test_contribution_scores_sequence(crested_object, keras_model, adata):
    sequence = "ATCGA" * 100
    (
        scores,
        one_hot_encoded_sequences,
    ) = crested_object.calculate_contribution_scores_sequence(
        sequence,
        class_names=list(adata.obs_names)[0:2],
        method="integrated_grad",
    )
    scores_refactored, one_hot_encoded_sequences_refactored = contribution_scores(
        input=sequence,
        model=keras_model,
        target_idx=[0, 1],
        method="integrated_grad",
    )
    assert np.allclose(
        scores,
        scores_refactored,
        atol=1e-5,
    ), "Scores are not equal."
    assert np.array_equal(
        one_hot_encoded_sequences,
        one_hot_encoded_sequences_refactored,
    ), "One-hot encoded sequences are not equal"


def test_contribution_scores_adata(crested_object, adata, keras_model, genome):
    scores, one_hot_encoded_sequences = crested_object.calculate_contribution_scores(
        class_names=list(adata.obs_names)[0:2],
        method="integrated_grad",
    )
    scores_refactored, one_hot_encoded_sequences_refactored = contribution_scores(
        input=adata,
        target_idx=[0, 1],
        model=keras_model,
        method="integrated_grad",
        genome=genome,
    )
    assert np.allclose(
        scores,
        scores_refactored,
        atol=1e-5,
    ), "Scores are not equal."
    assert np.array_equal(
        one_hot_encoded_sequences,
        one_hot_encoded_sequences_refactored,
    ), "One-hot encoded sequences are not equal"


def test_contribution_scores_modisco(
    crested_object_specific, adata_specific, keras_model, genome
):
    import shutil

    class_names = list(adata_specific.obs_names)[0:2]
    crested_object_specific.tfmodisco_calculate_and_save_contribution_scores(
        adata_specific,
        class_names=class_names,
        method="integrated_grad",
        output_dir="tests/data/test_contribution_scores",
    )
    # load scores and oh
    scores = np.load(
        f"tests/data/test_contribution_scores/{class_names[0]}_contrib.npz"
    )["arr_0"]
    one_hots = np.load(f"tests/data/test_contribution_scores/{class_names[0]}_oh.npz")[
        "arr_0"
    ]
    shutil.rmtree("tests/data/test_contribution_scores")  # ensure different outputs
    scores_ref_output, _ = contribution_scores_specific(
        input=adata_specific,
        target_idx=[0, 1],
        model=keras_model,
        method="integrated_grad",
        genome=genome,
        output_dir="tests/data/test_contribution_scores",
        transpose=True,
    )
    scores_refactored = np.load(
        f"tests/data/test_contribution_scores/{class_names[0]}_contrib.npz"
    )["arr_0"]
    one_hot_encoded_sequences_refactored = np.load(
        f"tests/data/test_contribution_scores/{class_names[0]}_oh.npz"
    )["arr_0"]
    print(scores.shape)
    print(scores_refactored.shape)
    assert np.allclose(
        scores,
        scores_refactored,
        atol=1e-5,
    ), "Scores are not equal."
    assert np.array_equal(
        one_hots,
        one_hot_encoded_sequences_refactored,
    ), "One-hot encoded sequences are not equal"
    assert np.allclose(
        scores_ref_output[:3, 0, :, :],
        scores,
        atol=1e-5,
    ), "Scores are not equal."
