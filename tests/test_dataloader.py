"""Test the dataloaders."""

import crested


def test_anndatawrapper(adata, genome):
    crested.pp.train_val_test_split(
        adata, strategy="region", val_size=0.1, test_size=0.1
    )

    datamodule = crested.tl.data.AnnDataWrapper(
        adata,
        genome=genome,
        batch_size=2,
        always_reverse_complement=True,
        max_stochastic_shift=3,
    )
    train_loader = datamodule.create_dataloader(split='train', augment=True, shuffle=True)
    val_loader = datamodule.create_dataloader(split='val')
    test_loader = datamodule.create_dataloader(split='test')
    predict_loader = datamodule.create_dataloader(split='predict')

    looping_backend(
        datamodule=datamodule,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        predict_loader=predict_loader,
    )

    # Try looping over the data

def test_anndatamodule(adata, genome):
    crested.pp.train_val_test_split(
        adata, strategy="region", val_size=0.1, test_size=0.1
    )

    datamodule = crested.tl.data.AnnDataModule(
        adata,
        genome=genome,
        batch_size=2,
        always_reverse_complement=True,
        max_stochastic_shift=3,
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    datamodule.setup("predict")

    looping_backend(
        datamodule=datamodule,
        train_loader=datamodule.train_dataloader.data,
        val_loader=datamodule.val_dataloader.data,
        test_loader=datamodule.test_dataloader.data,
        predict_loader=datamodule.predict_dataloader.data,
    )


def looping_backend(datamodule, train_loader, val_loader, test_loader, predict_loader):
    n_train_steps_per_epoch = datamodule.get_config()['n_train_steps_per_epoch']
    n_val_steps_per_epoch = datamodule.get_config()['n_val_steps_per_epoch']
    n_test_steps_per_epoch = datamodule.get_config()['n_test_steps_per_epoch']
    n_predict_steps_per_epoch = datamodule.get_config()['n_predict_steps_per_epoch']
    dataset_size = datamodule.get_config()['dataset_size']

    # Try looping over the data with lengths, like during fit()
    for _ in range(2):
        train_iter = iter(train_loader)
        for _ in range(n_train_steps_per_epoch):
            _, _ = next(train_iter)

    for _ in range(2):
        val_iter = iter(val_loader)
        for _ in range(n_val_steps_per_epoch):
            _, _ = next(val_iter)

    # Try looping over the data similar to keras tensorflow's predict(), catching end of iteration?
    for _ in range(2):
        test_iter = iter(test_loader)
        keep_looping = True
        i = 0
        while keep_looping:
            try:
                _, _ = next(test_iter)
                i += 1
            except StopIteration:
                keep_looping = False
            # Put in escape valve for infinite iteration
            if i > dataset_size:
                raise ValueError(f"Something went horribly wrong: looped over {i} batches while we only expected {n_test_steps_per_epoch}.")
    assert i == n_test_steps_per_epoch, "Did not loop over the expected number of samples"

    for _ in range(2):
        predict_iter = iter(predict_loader)
        keep_looping = True
        i = 0
        while keep_looping:
            try:
                _, _ = next(predict_iter)
                i += 1
            except StopIteration:
                keep_looping = False
            # Put in escape valve for infinite iteration
            if i > (dataset_size):
                raise ValueError(f"Something went horribly wrong: looped over {i} batches while we only expected {n_predict_steps_per_epoch}")
    assert i == n_predict_steps_per_epoch, "Did not loop over the expected number of samples"
