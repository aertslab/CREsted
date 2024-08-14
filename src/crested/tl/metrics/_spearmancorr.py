"""Spearman correlation metric."""

from __future__ import annotations

import keras


@keras.utils.register_keras_serializable(package="Metrics")
class SpearmanCorrelationPerClass(keras.metrics.Metric):
    """Spearman correlation metric per class."""

    def __init__(self, num_classes, name="spearman_correlation_per_class", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.correlation_sums = self.add_weight(
            name="correlation_sums", shape=(num_classes,), initializer="zeros"
        )
        self.update_counts = self.add_weight(
            name="update_counts", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        def _compute():
            return self.compute_correlation(y_true_non_zero, y_pred_non_zero)

        def _skip():
            return 0.0

        for i in range(self.num_classes):
            y_true_class = keras.ops.cast(y_true[:, i], dtype="float32")
            y_pred_class = keras.ops.cast(y_pred[:, i], dtype="float32")

            non_zero_indices = keras.ops.where(keras.ops.not_equal(y_true_class, 0))
            y_true_non_zero = keras.ops.take(y_true_class, non_zero_indices)
            y_pred_non_zero = keras.ops.take(y_pred_class, non_zero_indices)

            # Ensure sizes are constant by checking them before the operation
            num_elements = keras.ops.size(y_true_non_zero)
            proceed = num_elements > 1

            correlation = keras.ops.cond(proceed, _compute, _skip)
            self.correlation_sums[i].assign_add(correlation)
            self.update_counts[i].assign_add(keras.ops.cast(proceed, dtype="float32"))

    def compute_correlation(self, y_true_non_zero, y_pred_non_zero):
        ranks_true = keras.ops.argsort(keras.ops.argsort(y_true_non_zero))
        ranks_pred = keras.ops.argsort(keras.ops.argsort(y_pred_non_zero))

        rank_diffs = keras.ops.cast(ranks_true, dtype="float32") - keras.ops.cast(
            ranks_pred, dtype="float32"
        )
        rank_diffs_squared_sum = keras.ops.sum(keras.ops.square(rank_diffs))
        n = keras.ops.cast(keras.ops.size(y_true_non_zero), dtype="float32")

        correlation = 1 - (6 * rank_diffs_squared_sum) / (n * (n * n - 1))
        return keras.ops.where(keras.ops.isnan(correlation), 0.0, correlation)

    def result(self):
        valid_counts = self.update_counts
        avg_correlations = self.correlation_sums / valid_counts
        return keras.ops.mean(avg_correlations)

    def reset_state(self):
        self.correlation_sums.assign(keras.ops.zeros_like(self.correlation_sums))
        self.update_counts.assign(keras.ops.zeros_like(self.update_counts))
