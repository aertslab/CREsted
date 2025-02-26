import keras
import keras.ops as ops


@keras.saving.register_keras_serializable(package="Losses")
class PoissonMultinomialLoss(keras.losses.Loss):
    """
    Poisson decomposition with multinomial specificity term for aggregated counts.

    Combines Poisson loss for total counts with a multinomial term for class proportions.

    Parameters
    ----------
    total_weight
        Weight of the Poisson term in the total loss.
    eps
        Small value to avoid log(0).
    log_input
        If True, applies exponential transformation to predictions to produce counts.
    multinomial_axis
        Either "length" or "task", representing the axis along which multinomial proportions are calculated.
    reduction
        Type of reduction to apply to the loss: "mean" or "none".
    name
        Name of the loss function.
    """

    def __init__(
        self,
        total_weight: float = 1.0,
        eps: float = 1e-7,
        log_input: bool = True,
        multinomial_axis: str = "task",
        reduction: str = "sum_over_batch_size",
        name: str = "PoissonMultinomialLoss",
    ):
        """Initialize the PoissonMultinomialLoss."""
        super().__init__(name=name, reduction=reduction)
        self.total_weight = total_weight
        self.eps = eps
        self.log_input = log_input
        self.axis = 1 if multinomial_axis == "task" else 0

    def call(self, y_true, y_pred):
        """
        Compute the PoissonMultinomialLoss.

        Parameters
        ----------
        y_true
            True target values (aggregated counts).
        y_pred
            Predicted values.

        Returns
        -------
        Combined loss value.
        """
        # Ensure predictions and targets are float32
        y_true = ops.cast(y_true, dtype="float32")
        y_pred = ops.cast(y_pred, dtype="float32")

        # Apply exp if log_input is True
        if self.log_input:
            y_pred = ops.log(y_pred + 1)
            y_true = ops.log(y_true + 1)

        # Total counts along the specified axis
        total_true = ops.sum(y_true, axis=self.axis, keepdims=True)
        total_pred = ops.sum(y_pred, axis=self.axis, keepdims=True)

        # Poisson term
        poisson_term = total_pred - total_true * ops.log(total_pred + self.eps)

        # Multinomial probabilities
        p_pred = y_pred / (total_pred + self.eps)
        log_p_pred = ops.log(p_pred + self.eps)

        # Multinomial term
        multinomial_dot = -y_true * log_p_pred
        multinomial_term = ops.sum(multinomial_dot, axis=self.axis, keepdims=True)

        # Combine Poisson and Multinomial terms
        loss = multinomial_term + self.total_weight * poisson_term

        # Apply reduction
        if self.reduction == "mean":
            return ops.mean(loss)
        else:
            return loss

    def get_config(self):
        """Return the configuration of the loss function."""
        config = super().get_config()
        config.update(
            {
                "total_weight": self.total_weight,
                "eps": self.eps,
                "log_input": self.log_input,
                "axis": self.axis,
            }
        )
        return config
