"""Zero penalty metric."""

from __future__ import annotations

import keras


@keras.utils.register_keras_serializable(package="Metrics")
class ZeroPenaltyMetric(keras.metrics.Metric):
    """Zero penalty metric."""

    def __init__(self, name="zero_penalty_metric", **kwargs):
        """Initialize the metric."""
        super().__init__(name=name, **kwargs)
        self.zero_penalty = self.add_weight(name="zero_penalty", initializer="zeros")
        self.num_batches = self.add_weight(name="num_batches", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the state of the metric."""
        # Ensure y_true and y_pred are float32 for consistency
        y_true = keras.ops.cast(y_true, dtype="float32")
        y_pred = keras.ops.cast(y_pred, dtype="float32")

        # Create a mask for where y_true is zero
        zero_gt_mask = keras.ops.equal(y_true, 0)

        # Apply log transformation to positive and negative predictions
        log_y_pred_pos = keras.ops.log(1 + 1000 * keras.ops.maximum(y_pred, 0))
        log_y_pred_neg = -keras.ops.log(
            1 + keras.ops.abs(1000 * keras.ops.minimum(y_pred, 0))
        )

        # Combine the transformed positive and negative predictions
        log_y_pred = log_y_pred_pos + log_y_pred_neg

        # Calculate the zero penalty using the mask
        zero_penalty = keras.ops.sum(
            keras.ops.cast(zero_gt_mask, dtype="float32") * keras.ops.abs(log_y_pred)
        )

        # Update the state variables
        self.zero_penalty.assign_add(zero_penalty)
        self.num_batches.assign_add(1.0)

    def result(self):
        """Calculate the result of the metric by averaging the zero penalty over num batches."""
        return self.zero_penalty / self.num_batches

    def reset_state(self):
        """Reset the state of the metric."""
        self.zero_penalty.assign(0.0)
        self.num_batches.assign(0.0)
