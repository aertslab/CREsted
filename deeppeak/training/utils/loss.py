"""DeepPeak model loss functions."""
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.losses import Reduction


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, global_batch_size, reduction=Reduction.SUM, name="CustomLoss"):
        super().__init__(reduction=reduction, name=name)
        self.global_batch_size = global_batch_size

    @tf.function
    def call(self, y_true, y_pred):
        y_true_normalized = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=-1)
        cosine_loss = -tf.reduce_sum(y_true_normalized * y_pred_normalized, axis=-1)
        squared_difference_loss = K.mean(
            tf.math.squared_difference(y_pred, y_true), axis=-1
        )
        return (cosine_loss + squared_difference_loss) / self.global_batch_size

    def get_config(self):
        config = super().get_config()
        config.update({"global_batch_size": self.global_batch_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
