"""Spearman correlation metric."""

from __future__ import annotations
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Metrics")
class SpearmanCorrelationPerClass(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='spearman_correlation_per_class', **kwargs):
        super(SpearmanCorrelationPerClass, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.correlation_sums = self.add_weight(name='correlation_sums', shape=(num_classes,), initializer='zeros')
        self.update_counts = self.add_weight(name='update_counts', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_classes):
            y_true_class = tf.cast(y_true[:, i], tf.float32)
            y_pred_class = tf.cast(y_pred[:, i], tf.float32)

            non_zero_indices = tf.where(tf.not_equal(y_true_class, 0))
            y_true_non_zero = tf.gather(y_true_class, non_zero_indices)
            y_pred_non_zero = tf.gather(y_pred_class, non_zero_indices)

            # Ensure sizes are constant by checking them before the operation
            num_elements = tf.size(y_true_non_zero)
            proceed = num_elements > 1

            def compute():
                return self.compute_correlation(y_true_non_zero, y_pred_non_zero)

            def skip():
                return 0.0

            correlation = tf.cond(proceed, compute, skip)
            self.correlation_sums[i].assign_add(correlation)
            self.update_counts[i].assign_add(tf.cast(proceed, tf.float32))

    def compute_correlation(self, y_true_non_zero, y_pred_non_zero):
        ranks_true = tf.argsort(tf.argsort(y_true_non_zero))
        ranks_pred = tf.argsort(tf.argsort(y_pred_non_zero))
        
        rank_diffs = tf.cast(ranks_true, tf.float32) - tf.cast(ranks_pred, tf.float32)
        rank_diffs_squared_sum = tf.reduce_sum(tf.square(rank_diffs))
        n = tf.cast(tf.size(y_true_non_zero), tf.float32)
        
        correlation = 1 - (6 * rank_diffs_squared_sum) / (n * (n*n - 1))
        return tf.where(tf.math.is_nan(correlation), 0.0, correlation)

    def result(self):
        valid_counts = self.update_counts 
        avg_correlations = self.correlation_sums / valid_counts
        return tf.reduce_mean(avg_correlations)

    def reset_state(self):
        self.correlation_sums.assign(tf.zeros_like(self.correlation_sums))
        self.update_counts.assign(tf.zeros_like(self.update_counts))