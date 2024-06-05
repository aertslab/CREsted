from __future__ import annotations

import tensorflow as tf

from ._task import Task


class Classification(Task):
    def _setup_metrics(
        self, metrics: list[tf.keras.metrics.Metric] | None
    ) -> list[tf.keras.metrics.Metric]:
        default_metrics = [
            tf.keras.metrics.AUC(
                num_thresholds=200,
                curve="ROC",
                summation_method="interpolation",
                name="auROC",
                thresholds=None,
                multi_label=True,
                label_weights=None,
            ),
            tf.keras.metrics.AUC(
                num_thresholds=200,
                curve="PR",
                summation_method="interpolation",
                name="auPR",
                thresholds=None,
                multi_label=True,
                label_weights=None,
            ),
            tf.keras.metrics.CategoricalAccuracy(),
        ]

        if metrics is None:
            return default_metrics
        else:
            return metrics

    def _setup_loss(self, loss: tf.keras.losses.Loss | None) -> tf.keras.losses.Loss:
        if loss is None:
            return tf.keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            return loss
