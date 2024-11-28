import keras
import keras.ops as ops


@keras.saving.register_keras_serializable(package="Losses")
class PoissonLoss(keras.losses.Loss):
    """
    Custom Poisson loss for count data with optional log(x + 1) transformation.

    This loss function computes the Poisson loss, optionally applying
    log(x + 1) transformations to predictions and/or targets to ensure
    non-negativity.

    Parameters
    ----------
    log_transform : bool
        If True, applies log(x + 1) transformation to both predictions and targets.
    eps : float
        Small value to avoid log(0).
    reduction : str
        Type of reduction to apply to the loss. Default: "sum_over_batch_size".
    """

    def __init__(
        self,
        log_transform: bool = True,
        eps: float = 1e-7,
        reduction: str = "sum_over_batch_size",
        name: str = "PoissonLoss"
    ):
        """
        Initialize the PoissonLoss class.

        Parameters
        ----------
        log_transform : bool, optional
            Whether to apply a log(x + 1) transformation to the inputs. Default is True.
        eps : float, optional
            A small epsilon value to avoid log(0). Default is 1e-7.
        reduction : str, optional
            The type of reduction to apply to the loss, e.g., "sum_over_batch_size".
            Default is "sum_over_batch_size".
        name : str, optional
            Name of the loss function. Default is "PoissonLoss".
        """
        super().__init__(name=name, reduction=reduction)
        self.log_transform = log_transform
        self.eps = eps

    def call(self, y_true, y_pred):
        """
        Compute the Poisson loss.

        Parameters
        ----------
        y_true : Tensor
            True target values (counts or log(x + 1)-transformed counts).
        y_pred : Tensor
            Predicted values (counts or log(x + 1)-transformed counts).

        Returns
        -------
        Tensor
            The Poisson loss value for each sample.
        """
        # Ensure predictions and targets are float32
        y_true = ops.cast(y_true, dtype="float32")
        y_pred = ops.cast(y_pred, dtype="float32")

        # Apply log(x + 1) transformation if needed
        if self.log_transform:
            y_true = ops.log(y_true + 1.0)
            y_pred = ops.log(y_pred + 1.0)

        # Compute Poisson loss for each class
        loss = y_pred - y_true * ops.log(y_pred + self.eps)

        # Sum the loss across classes
        return ops.sum(loss, axis=-1)

    def get_config(self):
        """Return the configuration of the loss function."""
        config = super().get_config()
        config.update({
            "log_transform": self.log_transform,
            "eps": self.eps
        })
        return config
