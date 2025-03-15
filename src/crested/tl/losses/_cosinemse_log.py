"""CosineMSELogLoss class."""

from __future__ import annotations

import keras.ops


@keras.saving.register_keras_serializable(package="Losses")
class CosineMSELogLoss(keras.losses.Loss):
    """
    Custom loss function combining logarithmic transformation, cosine similarity, and mean squared error (MSE).

    This loss function applies a logarithmic transformation to predictions and true values,
    normalizes these values, and computes both MSE and cosine similarity. A dynamic weight
    based on the MSE is used to balance these two components.

    Parameters
    ----------
    max_weight
        The maximum weight applied to the cosine similarity loss component.
        Lower values will emphasize the MSE component, while higher values will emphasize the cosine similarity component.
    name
        Name of the loss function.
    reduction
        Type of reduction to apply to loss.
    multiplier
        Scalar to multiply the predicted value with. When predicting mean coverage, multiply by 1000 to get actual count. Keep to 1 when predicting insertion counts.

    Notes
    -----
    - The log transformation is `log(1 + 1000 * y)` for positive values and `-log(1 + abs(1000 * y))` for negative values.
    - The cosine similarity is computed between L2-normalized true and predicted values.
    - The dynamic weight for the cosine similarity component is constrained between 1.0 and `max_weight`.

    Examples
    --------
    >>> loss = CosineMSELogLoss(max_weight=2.0)
    >>> y_true = np.array([1.0, 0.0, -1.0])
    >>> y_pred = np.array([1.2, -0.1, -0.9])
    >>> loss(y_true, y_pred)
    """

    def __init__(
        self,
        max_weight: float = 1.0,
        name: str | None = "CosineMSELogLoss",
        reduction: str = "sum_over_batch_size",
        multiplier: float = 1000,
    ):
        """Initialize the loss function."""
        super().__init__(name=name)
        self.max_weight = max_weight
        self.reduction = reduction
        self.multiplier = multiplier

    def call(self, y_true, y_pred):
        """Compute the loss value."""
        y_true = keras.ops.cast(y_true, dtype="float32")
        y_pred = keras.ops.cast(y_pred, dtype="float32")

        y_true1 = keras.utils.normalize(y_true, axis=-1)
        y_pred1 = keras.utils.normalize(y_pred, axis=-1)

        log_y_pred_pos = keras.ops.log(
            1 + self.multiplier * keras.ops.maximum(y_pred, 0)
        )
        log_y_pred_neg = -keras.ops.log(
            1 + keras.ops.abs(self.multiplier * keras.ops.minimum(y_pred, 0))
        )

        log_y_pred = log_y_pred_pos + log_y_pred_neg
        log_y_true = keras.ops.log(1 + self.multiplier * y_true)

        mse_loss = keras.ops.mean(keras.ops.square(log_y_pred - log_y_true))
        weight = keras.ops.abs(mse_loss)
        weight = keras.ops.minimum(keras.ops.maximum(weight, 1.0), self.max_weight)

        cosine_loss = -keras.ops.sum(y_true1 * y_pred1, axis=-1)

        total_loss = weight * cosine_loss + mse_loss

        return total_loss

    def get_config(self):
        """Return the configuration of the loss function."""
        config = super().get_config()
        config.update({"max_weight": self.max_weight})
        return config

    @classmethod
    def from_config(cls, config):
        """Create a loss function from the configuration."""
        return cls(**config)
