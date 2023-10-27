"""DeepPeak model loss functions."""
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K


def custom_loss(y_true, y_pred):
    y_true1 = nn.l2_normalize(y_true, axis=-1)
    y_pred1 = nn.l2_normalize(y_pred, axis=-1)
    y_pred2 = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true2 = math_ops.cast(y_true, y_pred.dtype)
    return -math_ops.reduce_sum(y_true1 * y_pred1, axis=-1) + K.mean(
        math_ops.squared_difference(y_pred2, y_true2), axis=-1
    )
