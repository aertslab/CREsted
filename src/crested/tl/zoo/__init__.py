"""
Custom `tf.keras.Model` definitions that have shown to work well in specific use cases.

Supply these (or your own) to `tl.Crested(...)` to use them in training.
"""

from ._basenji import basenji
from ._borzoi import borzoi, borzoi_prime
from ._deeptopic_cnn import deeptopic_cnn
from ._deeptopic_lstm import deeptopic_lstm
from ._dilated_cnn import chrombpnet, dilated_cnn
from ._dilated_cnn_decoupled import chrombpnet_decoupled, dilated_cnn_decoupled
from ._enformer import enformer
from ._simple_convnet import simple_convnet
