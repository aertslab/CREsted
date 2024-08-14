from __future__ import annotations

import keras


@keras.utils.register_keras_serializable(package="Metrics")
class PearsonCorrelationLog(keras.metrics.Metric):
    """Log Pearson correlation metric."""

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

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure y_true and y_pred are float32 for consistency
        y_true = keras.ops.cast(y_true, dtype="float32")
        y_pred = keras.ops.cast(y_pred, dtype="float32")

        y_pred = keras.ops.where(y_pred < 0, keras.ops.zeros_like(y_pred), y_pred)

        y_true = keras.ops.log(y_true * 1000 + 1)
        y_pred = keras.ops.log(y_pred * 1000 + 1)

        self.y_true_sum.assign_add(keras.ops.sum(y_true))
        self.y_pred_sum.assign_add(keras.ops.sum(y_pred))
        self.y_true_squared_sum.assign_add(keras.ops.sum(keras.ops.square(y_true)))
        self.y_pred_squared_sum.assign_add(keras.ops.sum(keras.ops.square(y_pred)))
        self.y_true_y_pred_sum.assign_add(keras.ops.sum(y_true * y_pred))
        self.count.assign_add(keras.ops.cast(keras.ops.size(y_true), dtype="float32"))

    def result(self):
        numerator = (
            self.count * self.y_true_y_pred_sum - self.y_true_sum * self.y_pred_sum
        )
        denominator = keras.ops.sqrt(
            (self.count * self.y_true_squared_sum - keras.ops.square(self.y_true_sum))
            * (self.count * self.y_pred_squared_sum - keras.ops.square(self.y_pred_sum))
        )

        return numerator / (denominator + keras.backend.epsilon())

    def reset_state(self):
        self.y_true_sum.assign(0.0)
        self.y_pred_sum.assign(0.0)
        self.y_true_squared_sum.assign(0.0)
        self.y_pred_squared_sum.assign(0.0)
        self.y_true_y_pred_sum.assign(0.0)
        self.count.assign(0.0)
