from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Metrics")
class PearsonCorrelationLog(tf.keras.metrics.Metric):
    def __init__(self, name: str = "pearson_correlation_log", **kwargs):
        super().__init__(name=name, **kwargs)
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
        # Ensure y_true and y_pred are float32 for consistency
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_pred = tf.where(y_pred < 0, tf.zeros_like(y_pred), y_pred)

        y_true = tf.math.log(y_true * 1000 + 1)
        y_pred = tf.math.log(y_pred * 1000 + 1)

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
