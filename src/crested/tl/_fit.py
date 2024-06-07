from __future__ import annotations

import os
from datetime import datetime

import tensorflow as tf

from crested.tl import AnnDataLoader


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
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            **early_stopping_params
        )
        callbacks.append(early_stopping_callback)
    if model_checkpointing:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            **model_checkpointing_params
        )
        callbacks.append(model_checkpoint_callback)
    if learning_rate_reduce:
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
    train_data,
    val_data,
    project_name: str,
    epochs: int = 100,
    logger_type: str | None = "wandb",
    shuffle: bool = True,
    train_batch_size: int = 64,
    val_batch_size: int = 64,
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
    if model_checkpointing_params is None:
        model_checkpointing_params = {
            "filepath": os.path.join(project_name, "checkpoints", "{epoch:02d}.keras"),
            "monitor": "val_loss",
            "save_weights_only": False,
            "save_freq": "epoch",
            "save_best_only": True,
        }

    if learning_rate_reduce_params is None:
        learning_rate_reduce_params = {
            "monitor": "val_loss",
            "factor": 0.25,
            "patience": 4,
            "min_lr": 1e-6,
        }

    if early_stopping_params is None:
        early_stopping_params = {
            "patience": 10,
            "mode": "min",
            "monitor": "val_loss",
        }

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
        from wandb.integration.keras import WandbMetricsLogger

        now = datetime.now().strftime("%Y-%m-%d_%H:%M")

        logger = wandb.init(
            project=project_name,
            # entity='deep-lcb',
            name=now,
        )
        wandb_callback_epoch = WandbMetricsLogger(log_freq="epoch")
        wandb_callback_batch = WandbMetricsLogger(log_freq=10)

        callbacks.append(wandb_callback_epoch)
        callbacks.append(wandb_callback_batch)

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

    train_dataloader = AnnDataLoader(
        train_data, batch_size=train_batch_size, shuffle=True
    )
    val_dataloader = AnnDataLoader(val_data, batch_size=val_batch_size)

    # dataset_info = dataloader.info()

    n_train_steps_per_epoch = len(train_dataloader)
    n_val_steps_per_epoch = len(val_dataloader)

    # Log configs
    if logger_type == "wandb":
        wandb.config.update(
            {
                "project_name": project_name,
                "epochs": epochs,
                "train_batch_size": train_batch_size,
                "val_batch_size": val_batch_size,
                "n_train": len(train_data),
                "n_val": len(val_data),
                "n_train_steps_per_epoch": n_train_steps_per_epoch,
                "n_val_steps_per_epoch": n_val_steps_per_epoch,
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
    print(model.summary())

    model.fit(
        train_dataloader.data,
        validation_data=val_dataloader.data,
        epochs=epochs,
        steps_per_epoch=n_train_steps_per_epoch,
        validation_steps=n_val_steps_per_epoch,
        callbacks=callbacks,
    )

    if logger_type == "wandb":
        logger.finish()
