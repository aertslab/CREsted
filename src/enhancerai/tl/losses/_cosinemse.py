import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction
from tensorflow.python.keras import backend as K


class CosineMSELoss(Loss):
    def __init__(self, reduction=Reduction.SUM, name="CosineMSELoss"):
        super().__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        y_true_normalized = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=-1)
        cosine_loss = -tf.reduce_sum(y_true_normalized * y_pred_normalized, axis=-1)
        squared_difference_loss = K.mean(
            tf.math.squared_difference(y_pred, y_true), axis=-1
        )
        return cosine_loss + squared_difference_loss

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
