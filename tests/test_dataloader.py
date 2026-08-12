"""Test the dataloaders."""

import keras
import numpy as np
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


# Tests that MultiAnnDataWrapper loops correctly and returns one target array per AnnData
def test_multianndatawrapper(adata_preds, genome):
    datawrapper = crested.tl.data.MultiAnnDataWrapper(
        [adata_preds, adata_preds],
        genome=genome,
        batch_size=2,
        always_reverse_complement=True,
        max_stochastic_shift=3,
    )
    train_loader = datawrapper.create_dataloader(split='train', augment=True, shuffle=True)

    x, y = next(iter(train_loader))

    assert len(y) == 2, f"Expected one target per AnnData (2), but found {len(y)}."
    assert x.shape[0] == y[0].shape[0] == y[1].shape[0], "Batch size of input and both outputs is expected to be the same."
    np.testing.assert_array_equal(
        np.asarray(y[0]), np.asarray(y[1]),
        err_msg="Both AnnDatas are identical, so their targets should be identical too.",
    )


def test_multianndatawrapper_sizes(adata_preds, genome):
    datawrapper = crested.tl.data.MultiAnnDataWrapper(
        [adata_preds, adata_preds],
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
    assert datawrapper.get_config()['compressed'] == [False, False]


# Tests that a secondary AnnData with a different var_names order is correctly matched up by index rather than by position
def test_multianndatawrapper_reordered_indices(adata_preds, genome):
    regions = list(adata_preds.var_names)
    adata2 = create_anndata_with_regions(regions[::-1], n_classes=adata_preds.n_obs, random_state=1)

    datawrapper = crested.tl.data.MultiAnnDataWrapper(
        [adata_preds, adata2],
        genome=genome,
        batch_size=2,
    )

    region = regions[0]
    _, y = datawrapper.get_indexed_item(original_index=region)
    expected_1 = adata_preds.X[:, adata_preds.var_names.get_loc(region)].astype('float32')
    expected_2 = adata2.X[:, adata2.var_names.get_loc(region)].astype('float32')
    np.testing.assert_array_equal(y[0], expected_1)
    np.testing.assert_array_equal(y[1], expected_2)


# Tests that a secondary AnnData containing a superset of the first AnnData's regions is allowed and correctly indexed
def test_multianndatawrapper_superset_indices(adata_preds, genome):
    regions = list(adata_preds.var_names)
    extra_region = "chr1:1000000-1000500"
    adata2 = create_anndata_with_regions(
        regions + [extra_region], n_classes=adata_preds.n_obs, random_state=2
    )

    datawrapper = crested.tl.data.MultiAnnDataWrapper(
        [adata_preds, adata2],
        genome=genome,
        batch_size=2,
    )

    assert len(datawrapper.indices) == len(regions), "Indices should be based on the first AnnData, ignoring the second AnnData's extra region."

    region = regions[0]
    _, y = datawrapper.get_indexed_item(original_index=region)
    expected_2 = adata2.X[:, adata2.var_names.get_loc(region)].astype('float32')
    np.testing.assert_array_equal(y[1], expected_2)


# Tests that a secondary AnnData missing regions present in the first AnnData raises an error
def test_multianndatawrapper_missing_indices_raises(adata_preds, genome):
    regions = list(adata_preds.var_names)
    adata2 = create_anndata_with_regions(regions[:-1], n_classes=adata_preds.n_obs, random_state=3)

    with pytest.raises(AssertionError):
        crested.tl.data.MultiAnnDataWrapper(
            [adata_preds, adata2],
            genome=genome,
            batch_size=2,
        )

