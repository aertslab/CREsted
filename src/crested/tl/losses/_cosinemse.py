from __future__ import annotations

import keras


@keras.saving.register_keras_serializable(package="Losses")
class CosineMSELoss(keras.losses.Loss):
    """Custom loss function that combines cosine similarity and mean squared error."""

    def __init__(self, max_weight=1.0, reduction=None, name="CustomMSELoss"):
        super().__init__()
        self.max_weight = max_weight

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
