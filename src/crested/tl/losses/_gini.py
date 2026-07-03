# build custom loss
import keras


@keras.saving.register_keras_serializable()
class GiniLoss(keras.losses.Loss):
    """
    Custom loss that varies between MAE and MSE depending on the gini score of the ground truth.

    Parameters
    ----------
    starting_power
        The lowest power the error is scaled with, in case of gini=0. Gini=1 adds 1*increase_scale to this.
        If 1 (default) and increase_scale=1, will scale between MAE (power of 1) and MSE (power of 2), depending on Gini coefficient.
    increase_scale
        How much to increase the power per gini increase. 
        Default of 1 will add between 0 and 1 to the starting power, setting to 2 will add between 0 and 2 
        (making power scale from 1 to 3 with default starting_power), etc.
    name
        Name of the loss function.
    reduction
        Type of reduction to apply to loss.
    """

    def __init__(
        self,
        starting_power: float = 1.,
        increase_scale: float = 1.,
        name: str | None = "GiniLoss",
        reduction: str = "sum_over_batch_size",
    ):
        """Initialize the loss function"""
        super().__init__()
        self.starting_power = starting_power
        self.increase_scale = increase_scale
        self.reduction = reduction

    def gini_scores(self, array, eps = 0.0000001):
        """Calculate gini scores for the provided array"""
        # Loosely based on https://github.com/oliviaguest/gini/blob/master/gini.py

        # Values cannot be negative, so clip to 0
        array = keras.ops.maximum(array, 0)
        # Values cannot be 0:
        array = array + eps
        # Values must be sorted:
        array = keras.ops.sort(array, axis=1)
        # Number of array elements:
        n = array.shape[1]
        # Index per array element:
        index = keras.ops.expand_dims(keras.ops.arange(1, n+1, dtype='float32'), axis=0)
        # Gini coefficient:
        return ((keras.ops.sum((2 * index - n - 1) * array, axis=1)) / (n * keras.ops.sum(array, axis=1)))

    def call(self, y_true, y_pred):
        """Calculate the loss value."""
        # Calculate gini scores and add 1 to get range from 1 to 2
        power = self.starting_power + self.gini_scores(y_true) * self.increase_scale
        # Calculate absolute difference
        error = keras.ops.abs(y_true - y_pred)
        # Take power of, ranging from no change/absolute (power 1, gini 0, non-specific) and squared (power 2, gini 1, v specific)
        error = keras.ops.power(error, power[:, None])
        # Take mean
        error = keras.ops.mean(error, axis=1)
        return error
