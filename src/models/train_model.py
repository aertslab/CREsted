import click
import tensorflow as tf
import yaml
import os
from datetime import datetime
import atexit

import wandb
from wandb.keras import WandbMetricsLogger, WandbCallback

from dataloader import load_chunked_tfrecord_dataset, count_samples_in_tfrecords
from deeppeak.model import ChromBPNet
from deeppeak.metrics import get_lr_metric, PearsonCorrelation
from deeppeak.loss import custom_loss


def model_callbacks(checkpoint_dir: str, patience: int, use_wandb: bool) -> list:
    """Get model callbacks."""
    callbacks = []
    # Checkpoints
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, "{epoch:02d}"),
        save_freq="epoch",
        save_best_only=True,
    )
    callbacks.append(checkpoint)

    # Early stopping
    early_stop_metric = "val_loss"
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
        wandb_callback = WandbMetricsLogger(log_freq=10)
        wandb_model_callback = WandbCallback()
        callbacks.append(wandb_callback)
        callbacks.append(wandb_model_callback)

    return callbacks


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=True))
def main(input_dir: str, output_dir: str):
    # Init configs and wandb
    now = datetime.now().strftime("%Y-%m-%d_%H:%M")

    with open("configs/user.yml", "r") as f:
        config = yaml.safe_load(f)

    if config["wandb"]:
        run = wandb.init(
            project=f"deeppeak_{config['project_name']}",
            config=config,
            name=now,
        )

    checkpoint_dir = os.path.join(output_dir, config["project_name"], now)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Train on GPU
    strategy = tf.distribute.MirroredStrategy()
    atexit.register(strategy._extended._collective_ops._pool.close)
    gpus_found = tf.config.list_physical_devices("GPU")

    print("Number of replica devices in use: {}".format(strategy.num_replicas_in_sync))
    print("Number of GPUs available: {}".format(len(gpus_found)))

    if config["wandb"]:
        wandb.config.update({"num_gpus_available": len(gpus_found)})
        wandb.config.update({"num_devices_used": strategy.num_replicas_in_sync})

    # Load data
    fraction_of_data = config["fraction_of_data"]

    total_number_of_training_samples = count_samples_in_tfrecords(
        os.path.join(input_dir, "train", "*.tfrecord")
    )
    total_number_of_validation_samples = count_samples_in_tfrecords(
        os.path.join(input_dir, "val", "*.tfrecord")
    )
    total_number_of_test_samples = count_samples_in_tfrecords(
        os.path.join(input_dir, "test", "*.tfrecord")
    )
    if fraction_of_data < 1.0:
        # Testing
        print(f"WARNING: Using {fraction_of_data} of the data. ")
        total_number_of_training_samples = int(
            total_number_of_training_samples * fraction_of_data
        )
        total_number_of_validation_samples = int(
            total_number_of_validation_samples * fraction_of_data
        )
        total_number_of_test_samples = int(
            total_number_of_test_samples * fraction_of_data
        )

    if config["wandb"]:
        wandb.config.update(
            {
                "N_train": total_number_of_training_samples,
                "N_val": total_number_of_validation_samples,
                "N_test": total_number_of_test_samples,
            }
        )

    batch_size = config["batch_size"] * strategy.num_replicas_in_sync
    train = load_chunked_tfrecord_dataset(
        os.path.join(input_dir, "train", "*.tfrecord"),
        config,
        total_number_of_training_samples,
    )
    val = load_chunked_tfrecord_dataset(
        os.path.join(input_dir, "val", "*.tfrecord"),
        config,
        batch_size,
    )

    # Get one batch to check shapes
    for x, y in train.take(1):
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")

    train = strategy.experimental_distribute_dataset(train)
    val = strategy.experimental_distribute_dataset(val)

    # Initialize the model
    with strategy.scope():
        pt_model = config["pretrained_model_path"]

        if pt_model:
            print(f"Continuing training from pretrained model {pt_model}...")
            model = tf.keras.models.load_model(pt_model, compile=False)

        else:
            print("Training from scratch...")
            model = ChromBPNet(config)

        optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        lr_metric = get_lr_metric(optimizer)

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=custom_loss,
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanSquaredError(),
                tf.keras.metrics.CosineSimilarity(axis=1),
                PearsonCorrelation(),
                lr_metric,
            ],
        )

        callbacks = model_callbacks(checkpoint_dir, config["patience"], config["wandb"])
        output = model(tf.random.normal([batch_size, config["seq_len"], 4]))
        assert output.shape == (batch_size, config["num_classes"])

        train_steps_per_epoch = total_number_of_training_samples // batch_size
        val_steps_per_epoch = total_number_of_validation_samples // batch_size

        model.fit(
            train,
            steps_per_epoch=train_steps_per_epoch,
            validation_steps=val_steps_per_epoch,
            validation_data=val,
            epochs=config["epochs"],
            callbacks=callbacks,
        )

    loss, mae, rmse, cos, pearson, lr = model.evaluate(val, steps=val_steps_per_epoch)
    print(f"Validation Loss (MAE): {mae}")
    print(f"Validation Loss (RMSE): {rmse}")
    print(f"Validation Loss (Cosine Similarity): {cos}")
    print(f"Validation Loss (Pearson Correlation): {pearson}")
    print(f"Validation Loss (Learning Rate): {lr}")

    if config["wandb"]:
        run.finish()


if __name__ == "__main__":
    main()
