import tensorflow as tf
import yaml
import os
import shutil
import argparse
from datetime import datetime

import wandb
from wandb.keras import WandbMetricsLogger, WandbCallback

from models.zoo import simple_convnet, chrombpnet, basenji

from utils.metrics import get_lr_metric, PearsonCorrelation
from utils.loss import CustomLoss
from utils.augment import complement_base
from utils.dataloader import CustomDataset


def parse_arguments() -> argparse.Namespace:
    """Parse required command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "-g",
        "--genome_fasta_file",
        type=str,
        help="Path to genome FASTA file",
        default="data/raw/genome.fa",
    )
    parser.add_argument(
        "-b",
        "--bed_file",
        type=str,
        help="Path to BED file",
        default="data/processed/consensus_peaks_inputs.bed",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        type=str,
        help="Path to targets file",
        default="data/processed/targets.npz",
    )
    parser.add_argument(
        "-m",
        "--cell_mapping_file",
        type=str,
        help="Path to cell type mapping file",
        default="data/processed/cell_type_mapping.tsv",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Path to output directory",
        default="checkpoints/",
    )

    return parser.parse_args()


def model_callbacks(
    checkpoint_dir: str, patience: int, use_wandb: bool, profile: bool
) -> list:
    """Get model callbacks."""
    callbacks = []
    # Checkpoints
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, "checkpoints", "{epoch:02d}.keras"),
        save_freq="epoch",
        save_weights_only=False,
        save_best_only=False,
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
        wandb_callback_epoch = WandbMetricsLogger(log_freq="epoch")
        wandb_callback_batch = WandbMetricsLogger(log_freq=10)
        wandb_model_callback = WandbCallback()
        callbacks.append(wandb_callback_epoch)
        callbacks.append(wandb_callback_batch)
        callbacks.append(wandb_model_callback)

    if profile:
        print("Profiling enabled. Saving to logs/")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir="logs/", histogram_freq=1, profile_batch="10, 60"
        )
        callbacks.append(tensorboard_callback)

    return callbacks


def load_datasets(
    bed_file: str,
    genome_fasta_file: str,
    targets_file: str,
    config: dict,
    batch_size: int,
    checkpoint_dir: str,
):
    """Load train & val datasets."""
    # Load data
    split_dict = {"val": config["val"], "test": config["test"]}

    dataset = CustomDataset(
        bed_file,
        genome_fasta_file,
        targets_file,
        config["target"],
        split_dict,
        config["num_classes"],
        config["augment_shift_n_bp"],
        config["fraction_of_data"],
        checkpoint_dir,
    )

    seq_len = config["seq_len"]
    augment_complement = config["augment_complement"]

    base_to_int_mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(base_to_int_mapping.keys())),
            values=tf.constant(list(base_to_int_mapping.values()), dtype=tf.int32),
        ),
        default_value=-1,
    )

    @tf.function
    def mapped_function(sequence, target):
        if isinstance(sequence, str):
            sequence = tf.constant([sequence])
        elif isinstance(sequence, tf.Tensor) and sequence.ndim == 0:
            sequence = tf.expand_dims(sequence, 0)

        # Define one_hot_encode function using TensorFlow operations
        def one_hot_encode(sequence):
            # Map each base to an integer
            char_seq = tf.strings.unicode_split(sequence, "UTF-8")
            integer_seq = table.lookup(char_seq)
            # One-hot encode the integer sequence
            x = tf.one_hot(integer_seq, depth=4)
            if augment_complement:
                x = complement_base(x)
            return x

        # Apply one_hot_encode to each sequence
        one_hot_sequence = tf.map_fn(
            one_hot_encode,
            sequence,
            fn_output_signature=tf.TensorSpec(shape=(seq_len, 4), dtype=tf.float32),
        )
        one_hot_sequence = tf.squeeze(one_hot_sequence, axis=0)  # remove extra map dim
        return one_hot_sequence, target

    train_data = (
        dataset.subset("train")
        .map(mapped_function, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .repeat(config["epochs"])
        .prefetch(tf.data.AUTOTUNE)
    )

    val_data = (
        dataset.subset("val")
        .map(mapped_function, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .repeat(config["epochs"])
        .prefetch(tf.data.AUTOTUNE)
    )

    total_number_of_training_samples = dataset.len("train")
    total_number_of_validation_samples = dataset.len("val")

    return (
        train_data,
        val_data,
        total_number_of_training_samples,
        total_number_of_validation_samples,
    )


def load_model(config: dict) -> tf.keras.Model:
    """Load requested model from zoo using the given configuration"""
    model_name = config["model_architecture"]
    options = ["basenji", "chrombpnet", "simple_convnet"]
    if model_name not in options:
        raise ValueError(f"Model {model_name} not supported.")

    model_config = config[model_name]

    if model_name == "basenji":
        model = basenji(
            input_shape=(config["seq_len"], 4),
            output_shape=(1, config["num_classes"]),
            first_activation=model_config["first_activation"],
            activation=model_config["activation"],
            output_activation=model_config["output_activation"],
            first_filters=model_config["first_filters"],
            filters=model_config["filters"],
            first_kernel_size=model_config["first_kernel_size"],
            kernel_size=model_config["kernel_size"],
        )
    elif model_name == "chrombpnet":
        model = chrombpnet(
            input_shape=(config["seq_len"], 4),
            output_shape=(1, config["num_classes"]),
            first_conv_filters=model_config["first_conv_filters"],
            first_conv_filter_size=model_config["first_conv_filter_size"],
            first_conv_pool_size=model_config["first_conv_pool_size"],
            first_conv_activation=model_config["first_conv_activation"],
            first_conv_l2=model_config["first_conv_l2"],
            first_conv_dropout=model_config["first_conv_dropout"],
            n_dil_layers=model_config["n_dil_layers"],
            num_filters=model_config["num_filters"],
            filter_size=model_config["filter_size"],
            activation=model_config["activation"],
            l2=model_config["l2"],
            dropout=model_config["dropout"],
            batch_norm=model_config["batch_norm"],
            dense_bias=model_config["dense_bias"],
        )
    elif model_name == "simple_convnet":
        model = simple_convnet(
            input_shape=(config["seq_len"], 4),
            output_shape=(1, config["num_classes"]),
            num_conv_blocks=model_config["num_conv_blocks"],
            num_dense_blocks=model_config["num_dense_blocks"],
            residual=model_config["residual"],
            first_activation=model_config["first_activation"],
            activation=model_config["activation"],
            output_activation=model_config["output_activation"],
            first_filters=model_config["first_filters"],
            filters=model_config["filters"],
            first_kernel_size=model_config["first_kernel_size"],
            kernel_size=model_config["kernel_size"],
            first_pool_size=model_config["first_pool_size"],
            pool_size=model_config["pool_size"],
            conv_dropout=model_config["conv_dropout"],
            dense_dropout=model_config["dense_dropout"],
            flatten=model_config["flatten"],
            dense_size=model_config["dense_size"],
            bottleneck=model_config["bottleneck"],
        )
    return model


def main(args: argparse.Namespace, config: dict):
    # Init configs and wandb
    now = datetime.now().strftime("%Y-%m-%d_%H:%M")

    if config["wandb"]:
        run = wandb.init(
            project=f"deeppeak_{config['project_name']}",
            config=config,
            name=now,
        )
    if int(config["seed"]) > 0:
        tf.random.set_seed(int(config["seed"]))

    checkpoint_dir = os.path.join(args.output_dir, config["project_name"], now)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    elif os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)
    shutil.copyfile("configs/user.yml", os.path.join(checkpoint_dir, "user.yml"))
    shutil.copyfile(
        args.cell_mapping_file, os.path.join(checkpoint_dir, "cell_type_mapping.tsv")
    )
    exit()
    # Train on GPU
    gpus_found = tf.config.list_physical_devices("GPU")

    strategy = tf.distribute.MirroredStrategy()
    print("Number of replica devices in use: {}".format(strategy.num_replicas_in_sync))
    print("Number of GPUs available: {}".format(len(gpus_found)))

    if config["wandb"]:
        wandb.config.update({"num_gpus_available": len(gpus_found)})
        wandb.config.update({"num_devices_used": strategy.num_replicas_in_sync})

    # Mixed precision (for GPU)
    if config["mixed_precision"]:
        print("WARNING: Mixed precision enabled. Disable on CPU or older GPUs.")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Load data
    batch_size = config["batch_size"]
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    (
        train,
        val,
        total_number_of_training_samples,
        total_number_of_validation_samples,
    ) = load_datasets(
        args.bed_file,
        args.genome_fasta_file,
        args.targets_file,
        config,
        global_batch_size,
        checkpoint_dir,
    )

    if config["wandb"]:
        wandb.config.update(
            {
                "N_train": total_number_of_training_samples,
                "N_val": total_number_of_validation_samples,
            }
        )

    # Initialize the model
    with strategy.scope():
        pt_model = config["pretrained_model_path"]

        if pt_model:
            print(f"Continuing training from pretrained model {pt_model}...")
            model = tf.keras.models.load_model(
                pt_model,
                compile=True,
                custom_objects={
                    "lr": get_lr_metric,
                    "PearsonCorrelation": PearsonCorrelation,
                    "custom_loss": CustomLoss,
                },
            )

        else:
            print("Training from scratch...")
            model = load_model(config)

        optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        lr_metric = get_lr_metric(optimizer)

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=CustomLoss(global_batch_size),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanSquaredError(),
                tf.keras.metrics.CosineSimilarity(axis=1),
                PearsonCorrelation(),
                lr_metric,
            ],
        )

    # Get callbacks
    callbacks = model_callbacks(
        checkpoint_dir, config["patience"], config["wandb"], config["profile"]
    )

    # Train the model
    train_steps_per_epoch = total_number_of_training_samples // global_batch_size
    val_steps_per_epoch = total_number_of_validation_samples // global_batch_size

    model.fit(
        train,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        validation_data=val,
        epochs=config["epochs"],
        callbacks=callbacks,
    )

    if config["wandb"]:
        run.finish()


if __name__ == "__main__":
    # Load args and config
    args = parse_arguments()

    assert os.path.exists(
        "configs/user.yml"
    ), "users.yml file not found. Please run `make copyconfig first`"
    with open("configs/user.yml", "r") as f:
        config = yaml.safe_load(f)

    # Train
    main(args, config)
