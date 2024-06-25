from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.losses import Loss

class CosineMSELoss(Loss):
    """Custom loss function that combines cosine similarity and mean squared error."""

    def __init__(self, max_weight=1.0, name="CustomMSELoss", reduction=None):
        super().__init__(name=name)
        self.max_weight = max_weight
        self.reduction=reduction

    @tf.function
    def call(self, y_true, y_pred):
        # Ensure y_true and y_pred are float32 for consistency
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Normalize y_true and y_pred for cosine similarity
        y_true1 = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred1 = tf.nn.l2_normalize(y_pred, axis=-1)

        mse_loss = tf.reduce_mean(tf.square(y_pred - y_true))

        # Calculate dynamic weight (absolute value of MSE), with lower limit of 0 and an optional upper limit
        weight = tf.abs(mse_loss)
        weight = tf.minimum(
            tf.maximum(weight, 1.0), self.max_weight
        )  # Ensure weight does not exceed max_weight and minimum of 1.0

        # Calculate cosine similarity loss
        cosine_loss = -tf.reduce_sum(y_true1 * y_pred1, axis=-1)

        total_loss = weight * cosine_loss + mse_loss  

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_weight": self.max_weight,#})
            "reduction":self.reduction})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)