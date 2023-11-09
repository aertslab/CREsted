"""Custom metrics used in DeepPeak training."""

import tensorflow as tf


def get_lr_metric(optimizer):
    """Returns a function that gets the current learning rate from optimizer.
    Useful for logging the current learning rate during training.
    """

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


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

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        self.y_true_sum.assign_add(tf.reduce_sum(y_true))
        self.y_pred_sum.assign_add(tf.reduce_sum(y_pred))
        self.y_true_squared_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.y_pred_squared_sum.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.y_true_y_pred_sum.assign_add(tf.reduce_sum(y_true * y_pred))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        numerator = (
            self.count * self.y_true_y_pred_sum - self.y_true_sum * self.y_pred_sum
        )
        denominator = tf.sqrt(
            (self.count * self.y_true_squared_sum - tf.square(self.y_true_sum))
            * (self.count * self.y_pred_squared_sum - tf.square(self.y_pred_sum))
        )

        return numerator / (denominator + tf.keras.backend.epsilon())

    def reset_state(self):
        self.y_true_sum.assign(0.0)
        self.y_pred_sum.assign(0.0)
        self.y_true_squared_sum.assign(0.0)
        self.y_pred_squared_sum.assign(0.0)
        self.y_true_y_pred_sum.assign(0.0)
        self.count.assign(0.0)
