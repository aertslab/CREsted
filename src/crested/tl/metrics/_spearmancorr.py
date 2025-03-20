from keras import backend as K
from keras import metrics, ops
from keras.utils import register_keras_serializable


@register_keras_serializable(package="Metrics")
class SpearmanCorrelationPerClass(metrics.Metric):
    """Spearman correlation metric for multiclass models."""

    def __init__(self, num_classes, name="multiclass_spearman_correlation", **kwargs):
        """
        Initialize the metric.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Initialize weights for each class
        self.correlation_sums = [
            self.add_weight(name=f"class_{i}_correlation", initializer="zeros")
            for i in range(num_classes)
        ]
        self.counts = [
            self.add_weight(name=f"class_{i}_count", initializer="zeros")
            for i in range(num_classes)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the state of the metric.

        Args:
            y_true (tensor): Ground truth labels of shape (batch_size, num_classes).
            y_pred (tensor): Predicted outputs of shape (batch_size, num_classes).
        """
        y_true = ops.cast(y_true, dtype="float32")
        y_pred = ops.cast(y_pred, dtype="float32")

        for i in range(self.num_classes):
            # Extract the i-th class predictions and labels
            y_true_class = y_true[:, i]
            y_pred_class = y_pred[:, i]

            # Rank the predictions and true values
            y_true_rank = ops.argsort(ops.argsort(y_true_class))
            y_pred_rank = ops.argsort(ops.argsort(y_pred_class))

            # Convert ranks to float32 for calculations
            y_true_rank = ops.cast(y_true_rank, "float32")
            y_pred_rank = ops.cast(y_pred_rank, "float32")

            # Calculate numerator and denominator
            # Calculate numerator and denominator
            numerator = ops.cast(ops.size(y_true_class), dtype="float32") * ops.sum(
                y_true_rank * y_pred_rank
            ) - ops.sum(y_true_rank) * ops.sum(y_pred_rank)
            denominator = ops.sqrt(
                (
                    ops.cast(ops.size(y_true_class), dtype="float32")
                    * ops.sum(ops.square(y_true_rank))
                    - ops.square(ops.sum(y_true_rank))
                )
                * (
                    ops.cast(ops.size(y_true_class), dtype="float32")
                    * ops.sum(ops.square(y_pred_rank))
                    - ops.square(ops.sum(y_pred_rank))
                )
            )

            # Compute Spearman correlation for the class
            correlation = numerator / (denominator + K.epsilon())

            # Update running sum and count for the class
            self.correlation_sums[i].assign_add(correlation)
            self.counts[i].assign_add(1)

    def result(self):
        """Calculate the result of the metric."""
        # Compute the average Spearman correlation across all classes
        correlations = [
            self.correlation_sums[i] / (self.counts[i] + K.epsilon())
            for i in range(self.num_classes)
        ]
        return ops.mean(ops.stack(correlations))

    def reset_state(self):
        """Reset the state of the metric."""
        for i in range(self.num_classes):
            self.correlation_sums[i].assign(0.0)
            self.counts[i].assign(0.0)
