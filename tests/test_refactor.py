"""Test that ensures the outputs after the functional refactor are the same as before."""

import keras
import numpy as np
import pytest

from crested.tl import Crested, get_embeddings, predict, score_gene_locus
from crested.tl.data import AnnDataModule

np.random.seed(42)
keras.utils.set_random_seed(42)


@pytest.fixture(scope="module")
def crested_object(keras_model, adata, genome):
    anndatamodule = AnnDataModule(
        adata,
        genome_file=genome,
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


def test_get_embeddings(adata, crested_object, keras_model, genome):
    crested_object_embeddings = crested_object.get_embeddings(
        layer_name="denseblock_dense"
    )
    refactored_embeddings = get_embeddings(
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
        gene_locus=f"{chrom_name}:{gene_start}-{gene_end}",
        all_class_names=list(adata.obs_names),
        class_name=list(adata.obs_names)[0],
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
