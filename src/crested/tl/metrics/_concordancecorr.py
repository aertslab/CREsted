"""Concordance correlation metric."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Metrics")
class ConcordanceCorrelationCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name="concordance_correlation_coefficient", **kwargs):
        super().__init__(name=name, **kwargs)
        self.y_true_sum = self.add_weight(name="y_true_sum", initializer="zeros")
        self.y_pred_sum = self.add_weight(name="y_pred_sum", initializer="zeros")
        self.y_true_sq_sum = self.add_weight(name="y_true_sq_sum", initializer="zeros")
        self.y_pred_sq_sum = self.add_weight(name="y_pred_sq_sum", initializer="zeros")
        self.y_true_pred_sum = self.add_weight(
            name="y_true_pred_sum", initializer="zeros"
        )
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        batch_size = tf.cast(tf.size(y_true), tf.float32)

        self.y_true_sum.assign_add(tf.reduce_sum(y_true))
        self.y_pred_sum.assign_add(tf.reduce_sum(y_pred))
        self.y_true_sq_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.y_pred_sq_sum.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.y_true_pred_sum.assign_add(tf.reduce_sum(y_true * y_pred))
        self.count.assign_add(batch_size)

    def result(self):
        y_true_mean = tf.math.divide_no_nan(self.y_true_sum, self.count)
        y_pred_mean = tf.math.divide_no_nan(self.y_pred_sum, self.count)
        y_true_var = tf.math.divide_no_nan(self.y_true_sq_sum, self.count) - tf.square(
            y_true_mean
        )
        y_pred_var = tf.math.divide_no_nan(self.y_pred_sq_sum, self.count) - tf.square(
            y_pred_mean
        )
        covariance = (
            tf.math.divide_no_nan(self.y_true_pred_sum, self.count)
            - y_true_mean * y_pred_mean
        )

        numerator = 2 * covariance
        denominator = y_true_var + y_pred_var + tf.square(y_true_mean - y_pred_mean)

        return tf.math.divide_no_nan(
            numerator, denominator + tf.keras.backend.epsilon()
        )

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros_like(s))
