"""Concordance correlation metric."""

from __future__ import annotations

import keras


@keras.utils.register_keras_serializable(package="Metrics")
class ConcordanceCorrelationCoefficient(keras.metrics.Metric):
    """Concordance correlation coefficient metric."""

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
        y_true = keras.ops.cast(y_true, dtype="float32")
        y_pred = keras.ops.cast(y_pred, dtype="float32")
        batch_size = keras.ops.cast(keras.ops.size(y_true), dtype="float32")

        self.y_true_sum.assign_add(keras.ops.sum(y_true))
        self.y_pred_sum.assign_add(keras.ops.sum(y_pred))
        self.y_true_sq_sum.assign_add(keras.ops.sum(keras.ops.square(y_true)))
        self.y_pred_sq_sum.assign_add(keras.ops.sum(keras.ops.square(y_pred)))
        self.y_true_pred_sum.assign_add(keras.ops.sum(y_true * y_pred))
        self.count.assign_add(batch_size)

    def result(self):
        y_true_mean = keras.ops.divide_no_nan(self.y_true_sum, self.count)
        y_pred_mean = keras.ops.divide_no_nan(self.y_pred_sum, self.count)
        y_true_var = keras.ops.divide_no_nan(
            self.y_true_sq_sum, self.count
        ) - keras.ops.square(y_true_mean)
        y_pred_var = keras.ops.divide_no_nan(
            self.y_pred_sq_sum, self.count
        ) - keras.ops.square(y_pred_mean)
        covariance = (
            keras.ops.divide_no_nan(self.y_true_pred_sum, self.count)
            - y_true_mean * y_pred_mean
        )

        numerator = 2 * covariance
        denominator = (
            y_true_var + y_pred_var + keras.ops.square(y_true_mean - y_pred_mean)
        )

        return keras.ops.divide_no_nan(numerator, denominator + keras.backend.epsilon())

    def reset_state(self):
        for s in self.variables:
            s.assign(keras.ops.zeros_like(s))
