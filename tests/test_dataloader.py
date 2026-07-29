"""Test the dataloaders."""

import keras
import numpy as np
import pandas as pd
import pytest

import crested

from ._utils import create_anndata_with_regions


# Tests that splitting into splits at dataloader level works correctly, that it loops over data as expected, and that batching works correctly
def test_anndatawrapper(adata_preds, genome):
    datamodule = crested.tl.data.AnnDataWrapper(
        adata_preds,
        genome=genome,
        batch_size=2,
        always_reverse_complement=True,
        max_stochastic_shift=3,
    )
    train_loader = datamodule.create_dataloader(split='train', augment=True, shuffle=True)
    val_loader = datamodule.create_dataloader(split='val')
    test_loader = datamodule.create_dataloader(split='test')
    predict_loader = datamodule.create_dataloader(split='predict')

    if keras.config.backend() == 'tensorflow':
        looping_backend_tf(
            datamodule=datamodule,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            predict_loader=predict_loader,
        )
    elif keras.config.backend() == 'torch':
        looping_backend_pt(
            datamodule=datamodule,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            predict_loader=predict_loader,
        )


def test_anndatamodule(adata_preds, genome):
    datamodule = crested.tl.data.AnnDataModule(
        adata_preds,
        genome=genome,
        batch_size=2,
        always_reverse_complement=True,
        max_stochastic_shift=3,
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    datamodule.setup("predict")

    train_loader = datamodule.train_dataloader.data
    val_loader = datamodule.val_dataloader.data
    test_loader = datamodule.test_dataloader.data
    predict_loader = datamodule.predict_dataloader.data

    if keras.config.backend() == 'tensorflow':
        looping_backend_tf(
            datamodule=datamodule,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            predict_loader=predict_loader
        )
    elif keras.config.backend() == 'torch':
        looping_backend_pt(
            datamodule=datamodule,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            predict_loader=predict_loader
        )


# Test tensorflow iteration - works with an infinitely repeating generator and n_steps to know how many to query
def looping_backend_tf(datamodule, train_loader, val_loader, test_loader, predict_loader):
    n_train_steps_per_epoch = datamodule.get_config()['n_train_steps_per_epoch']
    n_val_steps_per_epoch = datamodule.get_config()['n_val_steps_per_epoch']
    n_test_steps_per_epoch = datamodule.get_config()['n_test_steps_per_epoch']
    n_predict_steps_per_epoch = datamodule.get_config()['n_predict_steps_per_epoch']

    # Try looping over the data with lengths, like during fit()
    for _ in range(2):
        split_sizes = {'train': 0, 'val': 0, 'test': 0, 'predict': 0}
        train_iter = iter(train_loader)
        for _ in range(n_train_steps_per_epoch):
            x, y = next(train_iter)
            split_sizes['train'] += x.shape[0]
        assert x.shape[0] == y.shape[0], "Batch size of input and output is expected to be the same."

        val_iter = iter(val_loader)
        for _ in range(n_val_steps_per_epoch):
            x, y = next(val_iter)
            split_sizes['val'] += x.shape[0]

        test_iter = iter(test_loader)
        for _ in range(n_test_steps_per_epoch):
            x, y = next(test_iter)
            split_sizes['test'] += x.shape[0]

        predict_iter = iter(predict_loader)
        for _ in range(n_predict_steps_per_epoch):
            x, y = next(predict_iter)
            split_sizes['predict'] += x.shape[0]

        for split_type in ['train', 'val', 'test', 'predict']:
            expected_size = datamodule.get_config()['n_'+split_type]
            assert split_sizes[split_type] == expected_size, f"Expected {split_type} dataset to be # of {split_type} samples ({expected_size}), but found {split_sizes[split_type]} samples."

### Test pytorch looping - works with integer indices and a range-based loop
def looping_backend_pt(datamodule, train_loader, val_loader, test_loader, predict_loader):
    for _ in range(2):
        split_sizes = {'train': 0, 'val': 0, 'test': 0, 'predict': 0}

        for x, _ in train_loader:
            split_sizes['train'] += x.shape[0]

        for x, y in val_loader:
            split_sizes['val'] += x.shape[0]
            assert x.shape[0] == y.shape[0], "Batch size of input and output is expected to be the same"

        for x, _ in test_loader:
            split_sizes['test'] += x.shape[0]

        for x, _ in predict_loader:
            split_sizes['predict'] += x.shape[0]

        for split_type in ['train', 'val', 'test', 'predict']:
            expected_size = datamodule.get_config()['n_'+split_type]
            assert split_sizes[split_type] == expected_size, f"Expected {split_type} dataset to be # of {split_type} samples ({expected_size}), but found {split_sizes[split_type]} samples."

# Tests whether the config split values (internal dataset sizes) match expected values.
# Especially important as we used the config split values as a ground truth above.
def test_anndatawrapper_sizes(adata_preds, genome):
    datawrapper = crested.tl.data.AnnDataWrapper(
        adata_preds,
        genome=genome,
        batch_size=2,
        always_reverse_complement=True,
        max_stochastic_shift=3,
    )

    # Dataset has 30 regions, 60% train, 20% val, 20% test
    # Train expected to be doubled given always_reverse_complement
    assert datawrapper.get_config()['n_train'] == (2*18), f"Expected 36 training samples (18 regions, rev-comp expanded), but found {datawrapper.get_config()['n_train']}"
    assert datawrapper.get_config()['n_val'] == 6, f"Expected 6 validation samples, but found {datawrapper.get_config()['n_val']}"
    assert datawrapper.get_config()['n_test'] == 6, f"Expected 6 test samples, but found {datawrapper.get_config()['n_test']}"
    assert datawrapper.get_config()['n_predict'] == 30, f"Expected 30 total samples, but found {datawrapper.get_config()['n_predict']}"

def test_anndatamodule_sizes(adata_preds, genome):
    datamodule = crested.tl.data.AnnDataModule(
        adata_preds,
        genome=genome,
        batch_size=2,
        always_reverse_complement=True,
        max_stochastic_shift=3,
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    datamodule.setup("predict")

    # Dataset has 30 regions, 60% train, 20% val, 20% test
    # Train expected to be doubled given always_reverse_complement
    assert datamodule.get_config()['n_train'] == (2*18), f"Expected 36 training samples (18 regions, rev-comp expanded), but found {datamodule.get_config()['n_train']}"
    assert datamodule.get_config()['n_val'] == 6, f"Expected 6 validation samples, but found {datamodule.get_config()['n_val']}"
    assert datamodule.get_config()['n_test'] == 6, f"Expected 6 test samples, but found {datamodule.get_config()['n_test']}"
    assert datamodule.get_config()['n_predict'] == 30, f"Expected 30 total samples, but found {datamodule.get_config()['n_predict']}"


@pytest.fixture
def adata_genes():
    """Anndata fixture with gene-body-shaped regions and a gene_id column."""
    regions = [
        "chr1:1000-2000:+",   # GENE_A: first gene on chrom (no upstream neighbor)
        "chr1:2010-3000:-",   # GENE_B: 10bp gap to GENE_A, 50bp gap to GENE_C
        "chr1:3050-4000:+",   # GENE_C: last gene on chrom (no downstream neighbor)
    ]
    adata = create_anndata_with_regions(regions, n_classes=4)
    adata.var["gene_id"] = ["GENE_A", "GENE_B", "GENE_C"]
    adata.var["split"] = ["train", "val", "test"]
    return adata


@pytest.fixture
def gene_neighbors_df():
    """Matching gene_neighbors table for `adata_genes`."""
    return pd.DataFrame({
        "name": ["GENE_A", "GENE_B", "GENE_C"],
        "prev_gene_end": [np.nan, 2000, 3000],
        "next_gene_start": [2010, 3050, np.nan],
    })


def test_gene_territory_mask_no_shift(adata_genes, gene_neighbors_df, genome):
    wrapper = crested.tl.data.AnnDataWrapper(
        adata_genes, genome=genome, gene_neighbors=gene_neighbors_df, gene_id_column="gene_id",
    )
    mask = wrapper._get_gene_territory_mask(
        original_index="chr1:1000-2000:+", parsed_index=("chr1", 1000, 2000, "+"),
    )
    assert mask.all()  # whole gene body is within [0, 2010)


def test_gene_territory_mask_shift_crosses_neighbor(adata_genes, gene_neighbors_df, genome):
    wrapper = crested.tl.data.AnnDataWrapper(
        adata_genes, genome=genome, gene_neighbors=gene_neighbors_df, gene_id_column="gene_id",
    )
    # Window becomes [1015, 2015); positions 2010..2014 (last 5) now belong to GENE_B's territory.
    mask = wrapper._get_gene_territory_mask(
        original_index="chr1:1000-2000:+", parsed_index=("chr1", 1000, 2000, "+"), shift=15,
    )
    expected = np.array([True] * 995 + [False] * 5)
    np.testing.assert_array_equal(mask, expected)


def test_gene_territory_mask_minus_strand_and_shift(adata_genes, gene_neighbors_df, genome):
    wrapper = crested.tl.data.AnnDataWrapper(
        adata_genes, genome=genome, gene_neighbors=gene_neighbors_df, gene_id_column="gene_id",
    )
    # GENE_B, window shifted to [1995, 2985); forward positions 1995..1999 (first 5) fall
    # before prev_gene_end=2000. Strand "-" reverses the mask, so those False values move to the end.
    mask = wrapper._get_gene_territory_mask(
        original_index="chr1:2010-3000:-", parsed_index=("chr1", 2010, 3000, "-"), shift=-15,
    )
    expected = np.array([True] * 985 + [False] * 5)
    np.testing.assert_array_equal(mask, expected)


def test_gene_territory_mask_revcomp_flag(adata_genes, gene_neighbors_df, genome):
    wrapper = crested.tl.data.AnnDataWrapper(
        adata_genes, genome=genome, gene_neighbors=gene_neighbors_df, gene_id_column="gene_id",
    )
    mask_fwd = wrapper._get_gene_territory_mask(
        original_index="chr1:1000-2000:+", parsed_index=("chr1", 1000, 2000, "+"), shift=15,
    )
    mask_revcomp = wrapper._get_gene_territory_mask(
        original_index="chr1:1000-2000:+", parsed_index=("chr1", 1000, 2000, "+"), shift=15, revcomp=True,
    )
    np.testing.assert_array_equal(mask_revcomp, mask_fwd[::-1])


def test_gene_territory_mask_missing_neighbor_unbounded(adata_genes, gene_neighbors_df, genome):
    wrapper = crested.tl.data.AnnDataWrapper(
        adata_genes, genome=genome, gene_neighbors=gene_neighbors_df, gene_id_column="gene_id",
    )
    # GENE_C has no downstream neighbor (NaN next_gene_start) -> falls back to chrom end, so a
    # large shift should not spuriously introduce False values.
    mask = wrapper._get_gene_territory_mask(
        original_index="chr1:3050-4000:+", parsed_index=("chr1", 3050, 4000, "+"), shift=100,
    )
    assert mask.all()


def test_anndatawrapper_gene_territory_channel_shape(adata_genes, gene_neighbors_df, genome):
    wrapper = crested.tl.data.AnnDataWrapper(
        adata_genes, genome=genome, gene_neighbors=gene_neighbors_df, gene_id_column="gene_id",
    )
    assert wrapper.input_shape[-1] == 5
    x, _ = wrapper[0]
    assert x.shape[-1] == 5
    assert np.all(x[:, 4] == 1.0)  # no stochastic shift by default -> whole window is in-territory


def test_anndatawrapper_no_gene_neighbors_unchanged_shape(adata_genes, genome):
    wrapper = crested.tl.data.AnnDataWrapper(adata_genes, genome=genome)
    assert wrapper.input_shape[-1] == 4  # regression: default behavior unaffected


def test_anndatawrapper_gene_neighbors_missing_column(adata_genes, gene_neighbors_df, genome):
    with pytest.raises(ValueError, match="gene_id_column"):
        crested.tl.data.AnnDataWrapper(
            adata_genes, genome=genome, gene_neighbors=gene_neighbors_df, gene_id_column="not_a_column",
        )


def test_anndatawrapper_gene_neighbors_missing_gene_id(adata_genes, genome):
    incomplete_df = pd.DataFrame({
        "name": ["GENE_A", "GENE_B"],  # missing GENE_C
        "prev_gene_end": [np.nan, 2000],
        "next_gene_start": [2010, 3050],
    })
    with pytest.raises(ValueError, match="GENE_C"):
        crested.tl.data.AnnDataWrapper(
            adata_genes, genome=genome, gene_neighbors=incomplete_df, gene_id_column="gene_id",
        )


def test_anndatawrapper_gene_neighbors_duplicate_name(adata_genes, genome):
    dup_df = pd.DataFrame({
        "name": ["GENE_A", "GENE_A", "GENE_B", "GENE_C"],
        "prev_gene_end": [np.nan, np.nan, 2000, 3000],
        "next_gene_start": [2010, 2010, 3050, np.nan],
    })
    with pytest.raises(ValueError, match="unique"):
        crested.tl.data.AnnDataWrapper(
            adata_genes, genome=genome, gene_neighbors=dup_df, gene_id_column="gene_id",
        )


def test_anndatawrapper_gene_neighbors_missing_required_columns(adata_genes, genome):
    bad_df = pd.DataFrame({"name": ["GENE_A", "GENE_B", "GENE_C"]})
    with pytest.raises(ValueError, match="required column"):
        crested.tl.data.AnnDataWrapper(
            adata_genes, genome=genome, gene_neighbors=bad_df, gene_id_column="gene_id",
        )
