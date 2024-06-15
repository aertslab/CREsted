"""Default task components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import tensorflow as tf

from crested.tl.losses import CosineMSELoss
from crested.tl.metrics import (
    ConcordanceCorrelationCoefficient,
    PearsonCorrelation,
    PearsonCorrelationLog,
    ZeroPenaltyMetric,
)


class BaseConfig(ABC):
    """Base configuration class for tasks."""

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
    """Default configuration for topic classification task."""

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


class PeakRegressionConfig(BaseConfig):
    """Default configuration for peak regression task."""

    @property
    def loss(self) -> tf.keras.losses.Loss:
        return CosineMSELoss()

    @property
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam(learning_rate=1e-3)

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.CosineSimilarity(axis=1),
            PearsonCorrelation(),
            ConcordanceCorrelationCoefficient(),
            PearsonCorrelationLog(),
            ZeroPenaltyMetric(),
        ]


class TaskConfig(NamedTuple):
    """
    Task configuration (optimizer, loss, and metrics) for use in tl.Crested.

    The TaskConfig class is a simple NamedTuple that holds the optimizer, loss, and metrics

    Example
    -------
    >>> optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    >>> loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    >>> metrics = [
    ...     tf.keras.metrics.AUC(
    ...         num_thresholds=200,
    ...         curve="ROC",
    ...     )
    ... ]
    >>> configs = TaskConfig(optimizer, loss, metrics)


    Attributes
    ----------
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer used for training.
    loss : tf.keras.losses.Loss
        Loss function used for training.
    metrics : list[tf.keras.metrics.Metric]
        Metrics used for training.
    """

    optimizer: tf.keras.optimizers.Optimizer
    loss: tf.keras.losses.Loss
    metrics: list[tf.keras.metrics.Metric]


def default_configs(
    task: str,
) -> TaskConfig:
    """
    Get default loss, optimizer, and metrics for an existing task.

    Possible tasks are:
    - "topic_classification"
    - "peak_regression"

    If what you want to do is not supported, you can create your own by using the TaskConfig class.

    Example
    -------
    >>> configs = default_configs("topic_classification")
    >>> optimizer, loss, metrics = configs.optimizer, configs.loss, configs.metrics
    >>> trainer = Crested(data, model, config=configs, project_name="test")

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
        "peak_regression": PeakRegressionConfig,
    }

    if task not in task_classes:
        raise ValueError(
            f"Task '{task}' not supported. Only {list(task_classes.keys())} are supported."
        )

    task_class = task_classes[task]()
    loss = task_class.loss
    optimizer = task_class.optimizer
    metrics = task_class.metrics

    return TaskConfig(optimizer=optimizer, loss=loss, metrics=metrics)
