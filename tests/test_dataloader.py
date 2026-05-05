"""Test the dataloaders."""

import keras

import crested


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

        for i in range(len(train_loader)):
            x, y = train_loader[i]
            split_sizes['train'] += x.shape[0]
        assert x.shape[0] == y.shape[0], "Batch size of input and output is expected to be the same"

        for i in range(len(val_loader)):
            x, y = val_loader[i]
            split_sizes['val'] += x.shape[0]

        for i in range(len(test_loader)):
            x, y = test_loader[i]
            split_sizes['test'] += x.shape[0]

        for i in range(len(predict_loader)):
            x, y = predict_loader[i]
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

