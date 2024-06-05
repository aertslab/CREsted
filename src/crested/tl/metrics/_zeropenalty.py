"""Zero penalty metric."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Metrics")
class ZeroPenaltyMetric(tf.keras.metrics.Metric):
    def __init__(self, name="zero_penalty_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.zero_penalty = self.add_weight(name="zero_penalty", initializer="zeros")
        self.num_batches = self.add_weight(name="num_batches", initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure y_true and y_pred are float32 for consistency
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        zero_gt_mask = tf.cast(tf.equal(y_true, 0), tf.float32)

        # Find indices of positive and negative predictions
        pos_indices = tf.where(tf.logical_and(y_pred >= 0, zero_gt_mask == 1))
        neg_indices = tf.where(tf.logical_and(y_pred < 0, zero_gt_mask == 1))

        # Process positive and negative y_pred values separately
        y_pred_pos = tf.gather_nd(y_pred, pos_indices)
        y_pred_neg = tf.gather_nd(y_pred, neg_indices)

        # Apply log transformation
        log_y_pred_pos = tf.math.log(1 + 1000 * y_pred_pos)  # Positive values as normal
        log_y_pred_neg = -tf.math.log(
            1 + tf.abs(1000 * y_pred_neg)
        )  # Absolute, log, then negate

        # Scatter back to original shape with zeros as placeholders
        log_y_pred = tf.scatter_nd(
            pos_indices, log_y_pred_pos, tf.shape(y_pred, out_type=tf.int64)
        ) + tf.scatter_nd(
            neg_indices, log_y_pred_neg, tf.shape(y_pred, out_type=tf.int64)
        )

        zero_penalty = tf.reduce_sum(zero_gt_mask * tf.abs(log_y_pred))
        self.zero_penalty.assign_add(zero_penalty)
        self.num_batches.assign_add(1.0)

    @tf.function
    def result(self):
        return self.zero_penalty / self.num_batches

    @tf.function
    def reset_state(self):
        self.zero_penalty.assign(0.0)
        self.num_batches.assign(0.0)
