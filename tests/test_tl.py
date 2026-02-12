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
    assert _transform_input(np.array([[[1, 0, 0, 0]]])).shape == (1, 1, 4)


def test_extract_layer_embeddings(keras_model, genome):
    input = "ATCGA" * 100
    embeddings = crested.tl.extract_layer_embeddings(input, keras_model, genome=genome, layer_name="denseblock_dense")
    assert embeddings.shape == (1, 8)
    input = ["ATCGA" * 100, "ATCGA" * 100]
    embeddings = crested.tl.extract_layer_embeddings(input, keras_model, layer_name="denseblock_dense")
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

    region_str = "chr1:1-501"
    region_str_pos = "chr1:1-501:+"
    region_str_neg = "chr1:1-501:-"
    predictions = crested.tl.predict(input=region_str, model=keras_model, genome=genome)
    predictions_pos = crested.tl.predict(input=region_str_pos, model=keras_model, genome=genome)
    predictions_neg = crested.tl.predict(input=region_str_neg, model=keras_model, genome=genome)
    assert predictions == pytest.approx(predictions_pos)
    assert predictions != pytest.approx(predictions_neg)

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
    assert os.path.exists(f"tests/data/test_contribution_scores/{class_names[0]}_contrib.npz")


def test_explainer_dtype_handling(keras_model):
    """Test that explainer functions handle non-float dtypes correctly.

    This test reproduces the bug reported in issue #138 where integrated_grad
    silently fails (returns 0) when given uint8 input instead of float32.
    Also tests saliency_map and smoothgrad for the same issue.
    """
    from crested.tl._explainer import integrated_grad, saliency_map, smoothgrad
    from crested.utils._seq_utils import one_hot_encode_sequence

    # Create a sequence that matches the model's expected input length
    expected_length = keras_model.input_shape[1]
    sequence = "ACGT" * (expected_length // 4)
    oh_float32 = one_hot_encode_sequence(sequence).astype(np.float32)
    oh_uint8 = one_hot_encode_sequence(sequence).astype(np.uint8)

    # Test integrated_grad with float32 (should work)
    result_float32 = integrated_grad(oh_float32, keras_model, class_index=1, num_baselines=2, num_steps=5)
    assert result_float32.sum() != 0, "integrated_grad should return non-zero values for float32 input"

    # Test integrated_grad with uint8 (currently fails silently, returning 0)
    result_uint8 = integrated_grad(oh_uint8, keras_model, class_index=1, num_baselines=2, num_steps=5)
    assert result_uint8.sum() != 0, "integrated_grad should not silently fail and return 0 for uint8 input"

    # The results should be similar (within reasonable tolerance)
    np.testing.assert_allclose(
        result_uint8,
        result_float32,
        rtol=1e-5,
        atol=1e-7,
        err_msg="integrated_grad should produce similar results for uint8 and float32 inputs",
    )

    # Test saliency_map with float32
    sal_float32 = saliency_map(oh_float32, keras_model, class_index=1, batch_size=128)
    assert sal_float32.sum() != 0, "saliency_map should return non-zero values for float32 input"

    # Test saliency_map with uint8
    sal_uint8 = saliency_map(oh_uint8, keras_model, class_index=1, batch_size=128)
    assert sal_uint8.sum() != 0, "saliency_map should not silently fail and return 0 for uint8 input"

    # The results should be similar
    np.testing.assert_allclose(
        sal_uint8,
        sal_float32,
        rtol=1e-5,
        atol=1e-7,
        err_msg="saliency_map should produce similar results for uint8 and float32 inputs",
    )

    # Test smoothgrad with float32
    smooth_float32 = smoothgrad(oh_float32, keras_model, class_index=1, num_samples=10)
    assert smooth_float32.sum() != 0, "smoothgrad should return non-zero values for float32 input"

    # Test smoothgrad with uint8
    smooth_uint8 = smoothgrad(oh_uint8, keras_model, class_index=1, num_samples=10)
    assert smooth_uint8.sum() != 0, "smoothgrad should not silently fail and return 0 for uint8 input"

    # Note: smoothgrad uses random noise, so we don't compare exact values between runs.
    # The important thing is that both produce non-zero results (i.e., the dtype fix works).


def test_enhancer_design_in_silico_evolution(keras_model, adata, genome):
    # one model
    seqs = crested.tl.design.in_silico_evolution(n_mutations=2, target=0, model=keras_model, n_sequences=1)
    assert len(seqs) == 1, len(seqs)
    assert len(seqs[0]) == keras_model.input_shape[1], len(seqs[0])

    # multiple models
    seqs = crested.tl.design.in_silico_evolution(
        n_mutations=2, target=1, model=[keras_model, keras_model], n_sequences=2
    )
    assert len(seqs) == 2, len(seqs)

    # acgt distribution provided
    acgt_disbtibution = crested.utils.calculate_nucleotide_distribution(input=adata, genome=genome, per_position=True)
    seqs = crested.tl.design.in_silico_evolution(
        n_mutations=1,
        target=0,
        model=keras_model,
        acgt_distribution=acgt_disbtibution,
    )

    # starting sequences provided
    starting_sequences = ["A" * keras_model.input_shape[1]]
    seqs = crested.tl.design.in_silico_evolution(
        n_mutations=1,
        target=0,
        model=keras_model,
        starting_sequences=starting_sequences,
    )
