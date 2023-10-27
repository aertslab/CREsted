import click
import tensorflow as tf
import numpy as np
import yaml
import os
from datetime import datetime

import wandb
from wandb.keras import WandbCallback

from deeppeak.model import DeepPeak
from deeppeak.metrics import get_lr_metric, PearsonCorrelation


def load_data(data_folder: str):
    X_train = np.load(os.path.join(data_folder, "X_train.npy"))
    X_val = np.load(os.path.join(data_folder, "X_val.npy"))
    y_train = np.load(os.path.join(data_folder, "Y_train.npy"))
    y_val = np.load(os.path.join(data_folder, "Y_val.npy"))

    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    X_val = tf.data.Dataset.from_tensor_slices(X_val)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    y_val = tf.data.Dataset.from_tensor_slices(y_val)

    return X_train, X_val, y_train, y_val


def model_callbacks(checkpoint_dir: str, patience: int, use_wandb: bool) -> list:
    """Get model callbacks."""
    callbacks = []
    # Checkpoints
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, "{epoch:02d}"),
        save_freq="epoch",
        save_weights_only=True,
    )
    callbacks.append(checkpoint)

    # Early stopping
    early_stop_metric = "val_pearson_correlation"
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=early_stop_metric, patience=patience, mode="max"
    )
    callbacks.append(early_stop)

    # Lr reduction
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=early_stop_metric,
        factor=0.25,
        patience=patience,
        min_lr=0.000001,
        mode="max",
    )
    callbacks.append(reduce_lr)

    # Wandb
    if use_wandb:
        wandb_callback = WandbCallback()
        callbacks.append(wandb_callback)

    return callbacks


# Assuming sequences of length 500 with 4 channels (e.g., DNA sequences: A, C, G, T)
# X_train = np.random.rand(1000, 500, 4)
# X_val = np.random.rand(300, 500, 4)

# # Assuming binary classification (adjust as needed for multi-class)
# y_train = np.random.rand(1000)
# y_val = np.random.rand(300)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=True))
def main(input_dir: str, output_dir: str):
    # Init configs and wandb
    now = datetime.now().strftime("%Y-%m-%d_%H:%M")

    with open("configs/user.yml", "r") as f:
        config = yaml.safe_load(f)

    if config["wandb"]:
        wandb.init(
            project=f"deeppeak_{config['project_name']}",
            config=config,
            name=now,
        )

    checkpoint_dir = os.path.join(output_dir, config["project_name"], now)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    X_train, X_val, y_train, y_val = load_data(input_dir)

    # Initialize the model
    model = DeepPeak(config["num_classes"], config)
    # model.build(input_shape=(None, 500, 4))

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    lr_metric = get_lr_metric(optimizer)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.CosineSimilarity(axis=1),
            PearsonCorrelation(),
            lr_metric,
        ],
    )

    callbacks = model_callbacks(checkpoint_dir, config["patience"], config["wandb"])

    model.fit(
        X_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    loss, mae, rmse, cos, pearson, lr = model.evaluate(X_val, y_val)
    print(f"Validation Loss (MAE): {mae}")
    print(f"Validation Loss (RMSE): {rmse}")
    print(f"Validation Loss (Cosine Similarity): {cos}")
    print(f"Validation Loss (Pearson Correlation): {pearson}")
    print(f"Validation Loss (Learning Rate): {lr}")


if __name__ == "__main__":
    main()
