"""Base class for all tasks in the tasks module."""

from __future__ import annotations

import tensorflow as tf


class Task:
    def __init__(
        self,
        loss: tf.keras.losses.Loss | None = None,
        optimizer: tf.keras.optimizers.Optimizer | None = None,
        learning_rate: float = 1e-3,
        metrics: list[tf.keras.metrics.Metric] | None = None,
    ):
        self.loss = self._setup_loss(loss)
        self.optimizer = self._setup_optimizer(optimizer, learning_rate)
        self.metrics = self._setup_metrics(metrics)

    def _setup_loss(self, loss: tf.keras.losses.Loss | None) -> tf.keras.losses.Loss:
        raise NotImplementedError("Subtasks should implement this method.")

    def _setup_optimizer(
        self,
        optimizer: tf.keras.optimizers.Optimizer | None,
        learning_rate: float,
    ) -> tf.keras.optimizers.Optimizer:
        if optimizer is None:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            return optimizer

    def _setup_metrics(
        self, metrics: list[tf.keras.metrics.Metric] | None
    ) -> list[tf.keras.metrics.Metric]:
        raise NotImplementedError("Subtasks should implement this method.")
