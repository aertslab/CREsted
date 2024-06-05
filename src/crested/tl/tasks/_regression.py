from __future__ import annotations

import tensorflow as tf

from crested.tl.losses import CosineMSELoss
from crested.tl.metrics import (
    ConcordanceCorrelationCoefficient,
    PearsonCorrelation,
    PearsonCorrelationLog,
    ZeroPenaltyMetric,
)

from ._task import Task


class Regression(Task):
    def _setup_metrics(
        self, metrics: list[tf.keras.metrics.Metric] | None
    ) -> list[tf.keras.metrics.Metric]:
        default_metrics = [
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.CosineSimilarity(axis=1),
            PearsonCorrelation(),
            ConcordanceCorrelationCoefficient(),
            PearsonCorrelationLog(),
            ZeroPenaltyMetric(),
        ]

        if metrics is None:
            return default_metrics
        else:
            return metrics

    def _setup_loss(self, loss: tf.keras.losses.Loss | None) -> tf.keras.losses.Loss:
        if loss is None:
            return CosineMSELoss()
        else:
            return loss
