"""
Custom `tf.Keras.losses.Loss` functions for specific use cases.

Supply these (or your own) to a `tl.TaskConfig` to be able to use them for training.
"""

from ._cosinemse import CosineMSELoss
from ._cosinemse_log import CosineMSELogLoss
from ._poisson import PoissonLoss
from ._poissonmultinomial import PoissonMultinomialLoss
