"""Default task components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import tensorflow as tf


class BaseConfig(ABC):
    @property
    @abstractmethod
    def loss(self) -> tf.keras.losses.Loss:
        pass

    @property
    @abstractmethod
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        pass

    @property
    @abstractmethod
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        pass


class TopicClassificationConfig(BaseConfig):
    @property
    def loss(self) -> tf.keras.losses.Loss:
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @property
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam(learning_rate=1e-3)

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
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


class ConfigComponents(NamedTuple):
    optimizer: tf.keras.optimizers.Optimizer
    loss: tf.keras.losses.Loss
    metrics: list[tf.keras.metrics.Metric]


def default_configs(
    task: str,
) -> tuple[
    tf.keras.optimizers.Optimizer, tf.keras.losses.Loss, list[tf.keras.metrics.Metric]
]:
    """Get default loss, optimizer, and metrics for a given task.

    Example
    -------
    >>> configs = default_configs("topic_classification")
    >>> optimizer, loss, metrics = configs.optimizer, configs.loss, configs.metrics

    Parameters
    ----------
    task : str
        Task for which to get default components.

    Returns
    -------
    tuple
        Optimizer, loss, and metrics for the given task.
    """
    task_classes = {
        "topic_classification": TopicClassificationConfig,
        # Add other tasks and their corresponding classes here
    }

    if task not in task_classes:
        raise ValueError(
            f"Task '{task}' not supported. Only {list(task_classes.keys())} are supported."
        )

    task_class = task_classes[task]()
    loss = task_class.loss
    optimizer = task_class.optimizer
    metrics = task_class.metrics

    return ConfigComponents(optimizer=optimizer, loss=loss, metrics=metrics)
