from typing import Literal

import keras
import keras.ops as ops
from loguru import logger


@keras.saving.register_keras_serializable(package="Losses")
class PoissonMultinomialLoss(keras.losses.Loss):
    """
    Poisson decomposition with multinomial specificity term for aggregated counts.

    Combines Poisson loss for total counts with a multinomial term for class proportions.

    Parameters
    ----------
    total_weight
        Weight of the Poisson term in the total loss. Increase to prioritize magnitude, decrease to prioritize shape.
    eps
        Small value to avoid log(0).
    log_transform
        Apply logarithmic transformation to truth and predictions.
    axis
        Axis to reduce over. If predicting 1D values, should be -1 (Decima-style). If predicting 2D values like tracks, can be -2 (to compare within tracks, like Borzoi) or -1 (to compare within classes, like Decima for every bin)
    norm
        How to normalize the data to [0,1] ranges for the multinomial loss. 'sum' (default) divides by the sum of that gene, 'max' divides by the max.
    reduction
        Type of reduction to apply to the loss: "mean" or "none".
    name
        Name of the loss function.
    """

    def __init__(
        self,
        total_weight: float = 1.0,
        log_transform: bool = True,
        axis: int = -1,
        eps: float = 1e-7,
        norm: Literal['sum', 'max'] = 'sum',
        reduction: str = "sum_over_batch_size",
        name: str = "PoissonMultinomialLoss",
        multinomial_axis = "deprecated",
        log_input = "deprecated",
    ):
        """Initialize the PoissonMultinomialLoss."""
        super().__init__(name=name, reduction=reduction)
        self.total_weight = total_weight
        self.eps = eps
        self.log_transform = log_transform
        self.norm = norm

        self.axis = axis
        if multinomial_axis != 'deprecated':
            raise ValueError("argument multinomial_axis is deprecated. Please use argument axis instead. Recommended replacements are axis=-1 or axis=2 for 'task', and axis=-2 or axis=1 for 'length'. Previous (bugged) bindings were axis=1 for 'task' and axis=0 for 'length'.")
        if log_input != "deprecated":
            logger.warning("log_input is renamed to log_transform, please use log_transform from now on.")
            self.log_transform = log_input


    def call(self, y_true, y_pred):
        """
        Compute the PoissonMultinomialLoss.

        Parameters
        ----------
        y_true
            True target values (aggregated counts).
        y_pred
            Predicted values.

        Returns
        -------
        Combined loss value.
        """
        # Ensure predictions and targets are float32
        y_true = ops.cast(y_true, dtype="float32")
        y_pred = ops.cast(y_pred, dtype="float32")

        # Apply exp if log_transform is True
        if self.log_transform:
            y_pred = ops.log1p(y_pred)
            y_true = ops.log1p(y_true)

        # Total counts along the specified axis
        total_true = ops.sum(y_true, axis=self.axis, keepdims=True)
        total_pred = ops.sum(y_pred, axis=self.axis, keepdims=True)

        # Poisson term
        poisson_term = total_pred - total_true * ops.log(total_pred + self.eps)

        # Multinomial probabilities
        if self.norm == 'sum':
            p_pred = y_pred / (total_pred + self.eps)
        elif self.norm == 'max':
            p_pred = y_pred / (ops.max(y_pred, axis=self.axis, keepdims=True) + self.eps)
        log_p_pred = ops.log(p_pred + self.eps)

        # Multinomial term
        multinomial_dot = -y_true * log_p_pred
        multinomial_term = ops.sum(multinomial_dot, axis=self.axis, keepdims=True)

        # Combine Poisson and Multinomial terms
        loss = multinomial_term + self.total_weight * poisson_term

        return loss

    def get_config(self):
        """Return the configuration of the loss function."""
        config = super().get_config()
        config.update(
            {
                "total_weight": self.total_weight,
                "eps": self.eps,
                "log_transform": self.log_transform,
                "axis": self.axis,
                "norm": self.norm,
            }
        )
        return config

    def __repr__(self):
        """Return a string representation of the loss."""
        return f"{self.name}: {self.get_config()}"


@keras.saving.register_keras_serializable(package="Losses")
class PoissonKLLoss(keras.losses.Loss):
    """
    Poisson decomposition with Kullback-Leibler specificity term for aggregated counts.

    Combines Poisson loss for total counts with a KL term for class proportions.

    Parameters
    ----------
    total_weight
        Weight of the Poisson term in the total loss. Increase to prioritize magnitude, decrease to prioritize shape.
    eps
        Small value to avoid log(0).
    log_transform
        Apply logarithmic transformation to truth and predictions.
    axis
        Axis to reduce over. If predicting 1D values, should be -1 (Decima-style). If predicting 2D values like tracks, can be -2 (to compare within tracks, like Borzoi) or -1 (to compare within classes, like Decima for every bin)
    norm
        How to normalize the data to [0,1] ranges for the KL loss. 'sum' (default) divides by the sum of that gene, 'max' divides by the max.
    reduction
        Type of reduction to apply to the loss: "mean" or "none".
    name
        Name of the loss function.
    """

    def __init__(
        self,
        total_weight: float = 1.0,
        log_transform: bool = True,
        axis: int = -1,
        eps: float = 1e-7,
        norm: Literal['sum', 'max'] = 'sum',
        reduction: str = "sum_over_batch_size",
        name: str = "PoissonKLLoss",
    ):
        """Initialize the PoissonKLLoss."""
        super().__init__(name=name, reduction=reduction)
        self.total_weight = total_weight
        self.eps = eps
        self.log_transform = log_transform
        self.norm = norm

        self.axis = axis


    def call(self, y_true, y_pred):
        """
        Compute the PoissonMultinomialLoss.

        Parameters
        ----------
        y_true
            True target values (aggregated counts).
        y_pred
            Predicted values.

        Returns
        -------
        Combined loss value.
        """
        # Ensure predictions and targets are float32
        y_true = ops.cast(y_true, dtype="float32") + self.eps
        y_pred = ops.cast(y_pred, dtype="float32") + self.eps

        # Apply exp if log_transform is True
        if self.log_transform:
            y_pred = ops.log1p(y_pred)
            y_true = ops.log1p(y_true)

        # Total counts along the specified axis
        total_true = ops.sum(y_true, axis=self.axis, keepdims=True)
        total_pred = ops.sum(y_pred, axis=self.axis, keepdims=True)

        # Poisson term
        poisson_term = total_pred - total_true * ops.log(total_pred)

        # Multinomial probabilities
        if self.norm == 'sum':
            p_true = y_true / (total_true)
            p_pred = y_pred / (total_pred)
        elif self.norm == 'max':
            p_true = y_true / (ops.max(y_true, axis=self.axis, keepdims=True))
            p_pred = y_pred / (ops.max(y_pred, axis=self.axis, keepdims=True))

        # KL term
        kl_term = y_true * ops.log(p_true / p_pred)

        # Combine Poisson and KL terms
        loss = kl_term + self.total_weight * poisson_term

        return loss

    def get_config(self):
        """Return the configuration of the loss function."""
        config = super().get_config()
        config.update(
            {
                "total_weight": self.total_weight,
                "eps": self.eps,
                "log_transform": self.log_transform,
                "axis": self.axis,
                "norm": self.norm,
            }
        )
        return config

    def __repr__(self):
        """Return a string representation of the loss."""
        return f"{self.name}: {self.get_config()}"
