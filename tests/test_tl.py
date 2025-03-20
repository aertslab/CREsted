"""Test tl module."""

import os

import numpy as np
import pytest

import crested
from crested.utils._utils import _detect_input_type, _transform_input


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
    assert _transform_input("AACGT").shape == (1, 5, 4), _transform_input("AACGT").shape
    assert np.array_equal(
        _transform_input("ACGT"),
        np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]),
    ), _transform_input("ACGT")
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


def test_extract_layer_embeddings(keras_model, genome):
    input = "ATCGA" * 100
    embeddings = crested.tl.extract_layer_embeddings(
        input, keras_model, genome=genome, layer_name="denseblock_dense"
    )
    assert embeddings.shape == (1, 8)
    input = ["ATCGA" * 100, "ATCGA" * 100]
    embeddings = crested.tl.extract_layer_embeddings(
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


def test_score_gene_locus(keras_model, genome):
    chrom = "chr1"
    start = 200000
    end = 200500
    scores, coordinates, min_loc, max_loc, tss_pos = crested.tl.score_gene_locus(
        chr_name=chrom,
        gene_start=start,
        gene_end=end,
        target_idx=1,
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


def test_contribution_scores(keras_model, genome):
    sequence = "ATCGA" * 100
    # test one sequence with multiple targets
    scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
        sequence,
        target_idx=[0, 1],
        model=keras_model,
        genome=genome,
        method="integrated_grad",
    )
    assert scores.shape == (1, 2, 500, 4)
    assert one_hot_encoded_sequences.shape == (1, 500, 4)

    sequences = ["ATCGA" * 100, "ATCGA" * 100]
    # test multiple sequences with one target
    scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
        sequences,
        target_idx=0,
        model=keras_model,
        genome=genome,
        method="integrated_grad",
    )
    assert scores.shape == (2, 1, 500, 4)
    assert one_hot_encoded_sequences.shape == (2, 500, 4)

    # test multiple models
    models = [keras_model, keras_model]
    scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
        sequence,
        target_idx=[0, 1],
        model=models,
        genome=genome,
        method="integrated_grad",
    )
    assert scores.shape == (1, 2, 500, 4)
    assert one_hot_encoded_sequences.shape == (1, 500, 4)

    # test batching sequences
    scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
        sequence,
        target_idx=1,
        model=keras_model,
        genome=genome,
        method="integrated_grad",
        batch_size=15,
    )
    assert scores.shape == (1, 1, 500, 4)
    assert one_hot_encoded_sequences.shape == (1, 500, 4)

    # test mutagenesis
    scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
        sequence,
        target_idx=1,
        model=models,
        genome=genome,
        method="mutagenesis",
    )
    assert scores.shape == (1, 1, 500, 4)
    assert one_hot_encoded_sequences.shape == (1, 500, 4)


def test_contribution_scores_specific(keras_model, adata, adata_specific, genome):
    with pytest.raises(ValueError):
        # class names can't be empty for specific
        crested.tl.contribution_scores_specific(
            input=adata_specific,
            target_idx=[],  # combined class
            model=keras_model,
            genome=genome,
            method="integrated_grad",
            transpose=True,
            verbose=False,
        )
    with pytest.raises(ValueError):
        # requires a specific anndata
        crested.tl.contribution_scores_specific(
            input=adata,
            target_idx=None,
            model=keras_model,
            genome=genome,
            method="integrated_grad",
            transpose=True,
            verbose=False,
        )
    scores, one_hots = crested.tl.contribution_scores_specific(
        input=adata_specific,
        target_idx=None,
        model=keras_model,
        genome=genome,
        method="integrated_grad",
        transpose=False,
        verbose=False,
    )
    assert scores.shape == (6, 1, 500, 4)
    assert one_hots.shape == (6, 500, 4)

    # test multiple models and subsetting class names
    scores, one_hots = crested.tl.contribution_scores_specific(
        input=adata_specific,
        target_idx=1,
        model=[keras_model, keras_model],
        genome=genome,
        method="integrated_grad",
        transpose=True,
        verbose=False,
    )
    assert scores.shape == (3, 1, 4, 500)
    assert one_hots.shape == (3, 4, 500)

    # test saving
    class_names = list(adata_specific.obs_names)
    scores, one_hots = crested.tl.contribution_scores_specific(
        input=adata_specific,
        target_idx=None,
        model=keras_model,
        genome=genome,
        method="integrated_grad",
        transpose=False,
        verbose=False,
        output_dir="tests/data/test_contribution_scores",
    )
    assert os.path.exists(
        f"tests/data/test_contribution_scores/{class_names[0]}_contrib.npz"
    )


def test_enhancer_design_in_silico_evolution(keras_model, adata, genome):
    # one model
    seqs = crested.tl.enhancer_design_in_silico_evolution(
        n_mutations=2, target=0, model=keras_model, n_sequences=1
    )
    assert len(seqs) == 1, len(seqs)
    assert len(seqs[0]) == keras_model.input_shape[1], len(seqs[0])

    # multiple models
    seqs = crested.tl.enhancer_design_in_silico_evolution(
        n_mutations=2, target=1, model=[keras_model, keras_model], n_sequences=2
    )
    assert len(seqs) == 2, len(seqs)

    # acgt distribution provided
    acgt_disbtibution = crested.utils.calculate_nucleotide_distribution(
        input=adata, genome=genome, per_position=True
    )
    seqs = crested.tl.enhancer_design_in_silico_evolution(
        n_mutations=1,
        target=0,
        model=keras_model,
        acgt_distribution=acgt_disbtibution,
    )

    # starting sequences provided
    starting_sequences = ["A" * keras_model.input_shape[1]]
    seqs = crested.tl.enhancer_design_in_silico_evolution(
        n_mutations=1,
        target=0,
        model=keras_model,
        starting_sequences=starting_sequences,
    )
