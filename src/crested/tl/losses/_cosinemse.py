from __future__ import annotations

import keras


@keras.saving.register_keras_serializable(package="Losses")
class CosineMSELoss(keras.losses.Loss):
    """
    Custom loss function that combines cosine similarity and mean squared error (MSE).

    This loss function computes both cosine similarity and MSE between the true and predicted values.
    A dynamic weight based on the MSE is used to balance these two components.

    Attributes
    ----------
    max_weight
        The maximum weight applied to the cosine similarity loss component.
        Lower values will emphasize the MSE component, while higher values will emphasize the cosine similarity component.
    reduction
        Type of reduction applied to loss.

    Parameters
    ----------
    max_weight
        The maximum weight applied to the cosine similarity loss component (default is 1.0).
        Lower values will emphasize the MSE component, while higher values will emphasize the cosine similarity component.
    name
        Name of the loss function.
    reduction
        Type of reduction to apply to loss.

    Notes
    -----
    - The cosine similarity is computed between L2-normalized true and predicted values.
    - The dynamic weight for the cosine similarity component is constrained between 1.0 and `max_weight`.

    Examples
    --------
    >>> loss = CosineMSELoss(max_weight=2.0)
    >>> y_true = np.array([1.0, 0.0, -1.0])
    >>> y_pred = np.array([1.2, -0.1, -0.9])
    >>> loss(y_true, y_pred)
    """

    def __init__(
        self,
        max_weight: float = 1.0,
        name: str | None = "CosineMSELoss",
        reduction: str = "sum_over_batch_size",
    ):
        super().__init__()
        self.max_weight = max_weight
        self.reduction = reduction

    def call(self, y_true, y_pred):
        # Ensure y_true and y_pred are float32 for consistency
        y_true = keras.ops.cast(y_true, dtype="float32")
        y_pred = keras.ops.cast(y_pred, dtype="float32")

        # Normalize y_true and y_pred for cosine similarity
        y_true1 = keras.utils.normalize(y_true, axis=-1)
        y_pred1 = keras.utils.normalize(y_pred, axis=-1)

        mse_loss = keras.ops.mean(keras.ops.square(y_pred - y_true))

        # Calculate dynamic weight (absolute value of MSE), with lower limit of 0 and an optional upper limit
        weight = keras.ops.abs(mse_loss)
        weight = keras.ops.minimum(
            keras.ops.maximum(weight, 1.0), self.max_weight
        )  # Ensure weight does not exceed max_weight and minimum of 1.0

        # Calculate cosine similarity loss
        cosine_loss = -keras.ops.sum(y_true1 * y_pred1, axis=-1)

        total_loss = weight * cosine_loss + mse_loss

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({"max_weight": self.max_weight, "reduction": self.reduction})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
