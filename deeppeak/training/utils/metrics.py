"""Custom metrics used in DeepPeak training."""

import tensorflow as tf
import numpy as np
import wandb

tf.keras.utils.get_custom_objects().clear()


def get_lr_metric(optimizer):
    """Returns a function that gets the current learning rate from optimizer.
    Useful for logging the current learning rate during training.
    """

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class LogMSEPerClassCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, class_names: list, val_steps: int):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.validation_steps = val_steps

    def on_epoch_end(self, epoch, logs=None):
        # Storage for predictions and labels
        predictions = []
        labels = []
        steps_done = 0

        # Iterate over one epoch of the validation data
        for x_val, y_val in self.validation_data:
            if steps_done == self.validation_steps:
                break
            preds = self.model.predict(x_val, verbose=0)
            predictions.extend(preds)
            labels.extend(y_val)
            steps_done += 1

        predictions = np.array(predictions)
        labels = np.array(labels)

        # Calculate MSE for each class
        mse_per_class = np.mean((predictions - labels) ** 2, axis=1)
        mae_per_class = np.mean(np.abs(predictions - labels), axis=1)

        log_data = {}
        for i, class_name in enumerate(self.class_names):
            log_data[f"celltype/mse/{class_name}"] = mse_per_class[i]
            log_data[f"celltype/mae/{class_name}"] = mae_per_class[i]

        # Log the MSE for each class to wandb
        wandb.log(log_data, commit=True)


@tf.keras.utils.register_keras_serializable(package="Metrics")
class PearsonCorrelation(tf.keras.metrics.Metric):
    def __init__(self, name="pearson_correlation", **kwargs):
        super(PearsonCorrelation, self).__init__(name=name, **kwargs)
        self.y_true_sum = self.add_weight(name="y_true_sum", initializer="zeros")
        self.y_pred_sum = self.add_weight(name="y_pred_sum", initializer="zeros")
        self.y_true_squared_sum = self.add_weight(
            name="y_true_squared_sum", initializer="zeros"
        )
        self.y_pred_squared_sum = self.add_weight(
            name="y_pred_squared_sum", initializer="zeros"
        )
        self.y_true_y_pred_sum = self.add_weight(
            name="y_true_y_pred_sum", initializer="zeros"
        )
        self.count = self.add_weight(name="count", initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        self.y_true_sum.assign_add(tf.reduce_sum(y_true))
        self.y_pred_sum.assign_add(tf.reduce_sum(y_pred))
        self.y_true_squared_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.y_pred_squared_sum.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.y_true_y_pred_sum.assign_add(tf.reduce_sum(y_true * y_pred))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    @tf.function
    def result(self):
        numerator = (
            self.count * self.y_true_y_pred_sum - self.y_true_sum * self.y_pred_sum
        )
        denominator = tf.sqrt(
            (self.count * self.y_true_squared_sum - tf.square(self.y_true_sum))
            * (self.count * self.y_pred_squared_sum - tf.square(self.y_pred_sum))
        )

        return numerator / (denominator + tf.keras.backend.epsilon())

    @tf.function
    def reset_state(self):
        self.y_true_sum.assign(0.0)
        self.y_pred_sum.assign(0.0)
        self.y_true_squared_sum.assign(0.0)
        self.y_pred_squared_sum.assign(0.0)
        self.y_true_y_pred_sum.assign(0.0)
        self.count.assign(0.0)
