"""Load trained tf model."""

import click
import tensorflow as tf
import numpy as np
import yaml
import os

from deeppeak.model import DeepPeak

# Assuming binary classification (adjust as needed for multi-class)
X_test = np.random.rand(300, 500, 4)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=True))
def main(input_dir: str, output_dir: str):
    """Load trained tf model."""

    with open("configs/user.yml", "r") as f:
        config = yaml.safe_load(f)

    # Load model
    num_classes = 19

    # Initialize the modelf
    model = DeepPeak(num_classes, config)
    model.build(input_shape=(None, 500, 4))

    latest = tf.train.latest_checkpoint("checkpoints")
    model.load_weights(latest).expect_partial()  # optimizer state not restored

    # Predict on test data
    X_test = np.random.rand(300, 500, 4)
    y_pred = model.predict(X_test)
    print("hello")
    print(y_pred)


if __name__ == "__main__":
    main()
