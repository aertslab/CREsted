"""Default task components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import keras

from crested.tl.losses import CosineMSELogLoss
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
    def loss(self) -> keras.losses.Loss:
        """Get default loss."""
        pass

    @property
    @abstractmethod
    def optimizer(self) -> keras.optimizers.Optimizer:
        """Get default optimizer."""
        pass

    @property
    @abstractmethod
    def metrics(self) -> list[keras.metrics.Metric]:
        """Get default metrics."""
        pass


class TopicClassificationConfig(BaseConfig):
    """Default configuration for topic classification task."""

    @property
    def loss(self) -> keras.losses.Loss:
        """Get default loss."""
        return keras.losses.BinaryCrossentropy(from_logits=False)

    @property
    def optimizer(self) -> keras.optimizers.Optimizer:
        """Get default optimizer."""
        return keras.optimizers.Adam(learning_rate=1e-3)

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        """Get default metrics."""
        return [
            keras.metrics.AUC(
                num_thresholds=200,
                curve="ROC",
                summation_method="interpolation",
                name="auROC",
                thresholds=None,
                multi_label=True,
                label_weights=None,
            ),
            keras.metrics.AUC(
                num_thresholds=200,
                curve="PR",
                summation_method="interpolation",
                name="auPR",
                thresholds=None,
                multi_label=True,
                label_weights=None,
            ),
            keras.metrics.CategoricalAccuracy(),
        ]


class PeakRegressionConfig(BaseConfig):
    """Default configuration for peak regression task."""

    def __init__(self, num_classes=None):
        """Initialize the configuration."""
        self.num_classes = num_classes

    @property
    def loss(self) -> keras.losses.Loss:
        """Get default loss."""
        return CosineMSELogLoss()

    @property
    def optimizer(self) -> keras.optimizers.Optimizer:
        """Get default optimizer."""
        return keras.optimizers.Adam(learning_rate=1e-3)

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        """Get default metrics."""
        metrics = [
            keras.metrics.MeanAbsoluteError(),
            keras.metrics.MeanSquaredError(),
            keras.metrics.CosineSimilarity(axis=1),
            PearsonCorrelation(),
            ConcordanceCorrelationCoefficient(),
            PearsonCorrelationLog(),
            ZeroPenaltyMetric(),
        ]
        return metrics


class TaskConfig(NamedTuple):
    """
    Task configuration (optimizer, loss, and metrics) for use in tl.Crested.

    The TaskConfig class is a simple NamedTuple that holds the optimizer, loss, and metrics

    Parameters
    ----------
    optimizer
        Optimizer used for training.
    loss
        Loss function used for training.
    metrics
        Metrics used for training.

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

    See Also
    --------
    crested.tl.default_configs
    """

    optimizer: keras.optimizers.Optimizer
    loss: keras.losses.Loss
    metrics: list[keras.metrics.Metric]

    def to_dict(self) -> dict:
        """
        Convert the TaskConfig to a dictionary.

        Useful for logging and saving the configuration.

        Returns
        -------
        Dictionary representation of the TaskConfig.
        """
        optimizer_info = {
            "optimizer": self.optimizer.__class__.__name__,
            "learning_rate": self.optimizer.learning_rate.numpy(),
        }
        loss_info = {"loss": self.loss.__class__.__name__}
        metrics_info = [metric.__class__.__name__ for metric in self.metrics]

        return {
            "optimizer": optimizer_info,
            "loss": loss_info,
            "metrics": metrics_info,
        }


def default_configs(task: str, num_classes: int | None = None) -> TaskConfig:
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
    tasks
        Task for which to get default components.
    num_classes
        Number of output classes of model. Required for Spearman correlation metric.

    Returns
    -------
    Optimizer, loss, and metrics for the given task.

    See Also
    --------
    crested.tl.TaskConfig
    """
    task_classes = {
        "topic_classification": TopicClassificationConfig,
        "peak_regression": PeakRegressionConfig,
    }

    if task not in task_classes:
        raise ValueError(
            f"Task '{task}' not supported. Only {list(task_classes.keys())} are supported."
        )

    task_class = (
        task_classes[task](num_classes=num_classes)
        if task == "peak_regression"
        else task_classes[task]()
    )
    loss = task_class.loss
    optimizer = task_class.optimizer
    metrics = task_class.metrics

    return TaskConfig(optimizer=optimizer, loss=loss, metrics=metrics)
