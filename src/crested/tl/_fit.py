from __future__ import annotations

import os
from datetime import datetime

import tensorflow as tf


def _initialize_callbacks(
    project_name: str,
    early_stopping: bool,
    early_stopping_params: dict | None,
    model_checkpointing: bool,
    model_checkpointing_params: dict | None,
    learning_rate_reduce: bool,
    learning_rate_reduce_params: dict | None,
    custom_callbacks: list | None,
) -> list:
    """Initialize callbacks"""
    callbacks = []
    if early_stopping:
        if early_stopping_params is None:
            early_stopping_params = {
                "patience": 10,
                "mode": "min",
                "monitor": "val/loss",
            }
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            **early_stopping_params
        )
        callbacks.append(early_stopping_callback)
    if model_checkpointing:
        if model_checkpointing_params is None:
            model_checkpointing_params = {
                "filename": os.path.join(
                    project_name, "checkpoints", "{epoch:02d}.keras"
                ),
                "monitor": "val/loss",
                "save_weights_only": False,
                "save_freq": "epoch",
                "save_best_only": True,
            }
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            **model_checkpointing_params
        )
        callbacks.append(model_checkpoint_callback)
    if learning_rate_reduce:
        if learning_rate_reduce_params is None:
            learning_rate_reduce_params = {
                "monitor": "val/loss",
                "factor": 0.25,
                "patience": 4,
                "min_lr": 1e-6,
            }
        learning_rate_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
            **learning_rate_reduce_params
        )
        callbacks.append(learning_rate_reduce_callback)
    if custom_callbacks is not None:
        callbacks.extend(custom_callbacks)
    return callbacks


def fit(
    task,
    model,
    dataloader,
    project_name: str,
    epochs: int = 100,
    logger_type: str | None = "wandb",
    shuffle: bool = True,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    random_reverse_complement: bool = False,
    always_reverse_complement: bool = True,
    mixed_precision: bool = False,
    model_checkpointing: bool = True,
    model_checkpointing_params: dict | None = None,
    early_stopping: bool = True,
    early_stopping_params: dict | None = None,
    learning_rate_reduce: bool = True,
    learning_rate_reduce_params: dict | None = None,
    custom_callbacks: list | None = None,
    seed: int | None = None,
):
    # Initialize callbacks
    callbacks = _initialize_callbacks(
        project_name,
        early_stopping,
        early_stopping_params,
        model_checkpointing,
        model_checkpointing_params,
        learning_rate_reduce,
        learning_rate_reduce_params,
        custom_callbacks,
    )

    # Initialize logger
    if logger_type == "wandb":
        import wandb

        now = datetime.now().strftime("%Y-%m-%d_%H:%M")

        logger = wandb.init(
            project=project_name,
            # entity='deep-lcb',
            name=now,
        )
    elif logger_type is None:
        logger = None
    else:
        raise ValueError(f"Invalid logger type: {logger_type}")

    # Fit model
    gpus_found = tf.config.list_physical_devices("GPU")

    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of replica devices in use: {strategy.num_replicas_in_sync}")
    print(f"Number of GPUs available: {len(gpus_found)}")

    if seed is not None:
        tf.random.set_seed(seed)

    if mixed_precision:
        print("WARNING: Mixed precision enabled. Disable on CPU or older GPUs.")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    dataloader.setup(
        stage="fit",
        random_reverse_complement=random_reverse_complement,
        always_reverse_complement=always_reverse_complement,
    )
    train_loader = dataloader.train_loader(batch_size=train_batch_size, shuffle=shuffle)
    val_loader = dataloader.val_loader(batch_size=val_batch_size, shuffle=False)

    dataset_info = dataloader.info()

    n_train_steps_per_epoch = dataset_info["n_train"] // train_batch_size
    n_val_steps_per_epoch = dataset_info["n_val"] // val_batch_size

    # Log configs
    if logger_type == "wandb":
        wandb.config.update(
            {
                "project_name": project_name,
                "epochs": epochs,
                "train_batch_size": train_batch_size,
                "val_batch_size": val_batch_size,
                "seq_len": dataset_info["seq_len"],
                "num_outputs": dataset_info["num_outputs"],
                "random_reverse_complement": random_reverse_complement,
                "always_reverse_complement": always_reverse_complement,
                "mixed_precision": mixed_precision,
                "model_checkpointing": model_checkpointing,
                "model_checkpointing_params": model_checkpointing_params,
                "early_stopping": early_stopping,
                "early_stopping_params": early_stopping_params,
                "learning_rate_reduce": learning_rate_reduce,
                "learning_rate_reduce_params": learning_rate_reduce_params,
                "custom_callbacks": custom_callbacks,
                "seed": seed,
            }
        )

    # compile and fit
    with strategy.scope():
        model.compile(
            optimizer=task.optimizer,
            loss=task.loss,
            metrics=task.metrics,
        )

    model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=epochs,
        steps_per_epoch=n_train_steps_per_epoch,
        validation_steps=n_val_steps_per_epoch,
        callbacks=callbacks,
    )

    if logger_type == "wandb":
        logger.finish()
