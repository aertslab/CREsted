"""Combine often used metrics."""

from __future__ import annotations

import tensorflow as tf


def metrics_collection(task: str) -> list[tf.keras.metrics.Metric]:
    """Returns a list of metrics based on the specified task."""
    if task == "topic_classification":
        return [
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
    elif task == "peak_regression":
        return []
    else:
        raise ValueError(f"Unknown task: {task}")
