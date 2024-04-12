"""DeepPeak model loss functions."""
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.losses import Reduction
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.losses import Loss


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
        return (cosine_loss + squared_difference_loss) #/ self.global_batch_size

    def get_config(self):
        config = super().get_config()
        config.update({"global_batch_size": self.global_batch_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class CustomLossV2(Loss):
    def __init__(self, max_weight=1.0, name="CustomLossV2"):
        super().__init__(name=name)
        self.max_weight = max_weight

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
        weight = tf.minimum(tf.maximum(weight, 1.0), self.max_weight)  # Ensure weight does not exceed max_weight and minimum of 1.0

        # Calculate cosine similarity loss
        cosine_loss = -tf.reduce_sum(y_true1 * y_pred1, axis=-1)
        ## Penalty for non-zero predictions when GT is zero
        #zero_gt_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
        #zero_penalty = tf.reduce_mean(zero_gt_mask * tf.abs(y_pred))
        #scaled_zero_penalty = 1 * zero_penalty  # Scale the penalty
        total_loss = weight * cosine_loss + mse_loss #+ scaled_zero_penalty

        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_weight": self.max_weight})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
class CustomLossMSELogV2_(Loss):
    def __init__(self, max_weight=1.0, name="CustomLossMSELogV2"):
        super().__init__(name=name)
        self.max_weight = max_weight

    @tf.function
    def call(self, y_true, y_pred):
        # Ensure y_true and y_pred are float32 for consistency
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Normalize y_true and y_pred for cosine similarity
        y_true1 = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred1 = tf.nn.l2_normalize(y_pred, axis=-1)

        # Find indices of positive and negative predictions
        pos_indices = tf.where(y_pred >= 0)
        neg_indices = tf.where(y_pred < 0)

        # Process positive and negative y_pred values separately
        y_pred_pos = tf.gather_nd(y_pred, pos_indices)
        y_pred_neg = tf.gather_nd(y_pred, neg_indices)

        # Apply log transformation
        log_y_pred_pos = tf.math.log(1 + 1000 * y_pred_pos)  # Positive values as normal
        log_y_pred_neg = -tf.math.log(1 + tf.abs(1000 * y_pred_neg))  # Absolute, log, then negate

        # Scatter back to original shape with zeros as placeholders
        log_y_pred = tf.scatter_nd(pos_indices, log_y_pred_pos, tf.shape(y_pred, out_type=tf.int64)) + \
                     tf.scatter_nd(neg_indices, log_y_pred_neg, tf.shape(y_pred, out_type=tf.int64))

        # Apply log transformation to y_true (ensure non-negativity as mentioned)
        log_y_true = tf.math.log(1 + 1000 * y_true)
        mse_loss = tf.reduce_mean(tf.square(log_y_pred - log_y_true))

        # Calculate dynamic weight (absolute value of MSE), with lower limit of 0 and an optional upper limit
        weight = tf.abs(mse_loss)
        weight = tf.minimum(tf.maximum(weight, 1.0), self.max_weight)  # Ensure weight does not exceed max_weight and minimum of 1.0

        # Calculate cosine similarity loss
        cosine_loss = -tf.reduce_sum(y_true1 * y_pred1, axis=-1)
        ## Penalty for non-zero predictions when GT is zero
        #zero_gt_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
        #zero_penalty = tf.reduce_sum(zero_gt_mask * tf.abs(log_y_pred))
        #scaled_zero_penalty = 0.001 * zero_penalty  # Scale the penalty
        total_loss = weight * cosine_loss + mse_loss #+ scaled_zero_penalty

        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_weight": self.max_weight})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
