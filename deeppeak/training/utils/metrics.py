"""Custom metrics used in DeepPeak training."""

import tensorflow as tf
import numpy as np
import wandb

tf.keras.utils.get_custom_objects().clear()


def get_lr_metric(optimizer):
    """Returns a function that gets the current learning rate from optimizer.
    Useful for logging the current learning rate during training.
    """

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class LogMSEPerClassCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, class_names: list, val_steps: int):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.validation_steps = val_steps

    def on_epoch_end(self, epoch, logs=None):
        # Storage for predictions and labels
        predictions = []
        labels = []
        steps_done = 0

        # Iterate over one epoch of the validation data
        for x_val, y_val in self.validation_data:
            if steps_done == self.validation_steps:
                break
            preds = self.model.predict(x_val, verbose=0)
            predictions.extend(preds)
            labels.extend(y_val)
            steps_done += 1

        predictions = np.array(predictions)
        labels = np.array(labels)

        # Calculate MSE for each class
        mse_per_class = np.mean((predictions - labels) ** 2, axis=1)
        mae_per_class = np.mean(np.abs(predictions - labels), axis=1)

        log_data = {}
        for i, class_name in enumerate(self.class_names):
            log_data[f"celltype/mse/{class_name}"] = mse_per_class[i]
            log_data[f"celltype/mae/{class_name}"] = mae_per_class[i]

        # Log the MSE for each class to wandb
        wandb.log(log_data, commit=True)


@tf.keras.utils.register_keras_serializable(package="Metrics")
class PearsonCorrelation(tf.keras.metrics.Metric):
    def __init__(self, name="pearson_correlation", **kwargs):
        super(PearsonCorrelation, self).__init__(name=name, **kwargs)
        self.y_true_sum = self.add_weight(name="y_true_sum", initializer="zeros")
        self.y_pred_sum = self.add_weight(name="y_pred_sum", initializer="zeros")
        self.y_true_squared_sum = self.add_weight(
            name="y_true_squared_sum", initializer="zeros"
        )
        self.y_pred_squared_sum = self.add_weight(
            name="y_pred_squared_sum", initializer="zeros"
        )
        self.y_true_y_pred_sum = self.add_weight(
            name="y_true_y_pred_sum", initializer="zeros"
        )
        self.count = self.add_weight(name="count", initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        self.y_true_sum.assign_add(tf.reduce_sum(y_true))
        self.y_pred_sum.assign_add(tf.reduce_sum(y_pred))
        self.y_true_squared_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.y_pred_squared_sum.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.y_true_y_pred_sum.assign_add(tf.reduce_sum(y_true * y_pred))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    @tf.function
    def result(self):
        numerator = (
            self.count * self.y_true_y_pred_sum - self.y_true_sum * self.y_pred_sum
        )
        denominator = tf.sqrt(
            (self.count * self.y_true_squared_sum - tf.square(self.y_true_sum))
            * (self.count * self.y_pred_squared_sum - tf.square(self.y_pred_sum))
        )

        return numerator / (denominator + tf.keras.backend.epsilon())

    @tf.function
    def reset_state(self):
        self.y_true_sum.assign(0.0)
        self.y_pred_sum.assign(0.0)
        self.y_true_squared_sum.assign(0.0)
        self.y_pred_squared_sum.assign(0.0)
        self.y_true_y_pred_sum.assign(0.0)
        self.count.assign(0.0)

class ZeroPenaltyMetric(tf.keras.metrics.Metric):
    def __init__(self, name='zero_penalty_metric', **kwargs):
        super(ZeroPenaltyMetric, self).__init__(name=name, **kwargs)
        self.zero_penalty = self.add_weight(name='zero_penalty', initializer='zeros')
        self.num_batches = self.add_weight(name='num_batches', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure y_true and y_pred are float32 for consistency
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        zero_gt_mask = tf.cast(tf.equal(y_true, 0), tf.float32)

        # Find indices of positive and negative predictions
        pos_indices = tf.where(tf.logical_and(y_pred >= 0, zero_gt_mask == 1))
        neg_indices = tf.where(tf.logical_and(y_pred < 0, zero_gt_mask == 1))

        # Process positive and negative y_pred values separately
        y_pred_pos = tf.gather_nd(y_pred, pos_indices)
        y_pred_neg = tf.gather_nd(y_pred, neg_indices)

        # Apply log transformation
        log_y_pred_pos = tf.math.log(1 + 1000 * y_pred_pos)  # Positive values as normal
        log_y_pred_neg = -tf.math.log(1 + tf.abs(1000 * y_pred_neg))  # Absolute, log, then negate

        # Scatter back to original shape with zeros as placeholders
        log_y_pred = tf.scatter_nd(pos_indices, log_y_pred_pos, tf.shape(y_pred, out_type=tf.int64)) + \
                     tf.scatter_nd(neg_indices, log_y_pred_neg, tf.shape(y_pred, out_type=tf.int64))

        zero_penalty = tf.reduce_sum(zero_gt_mask * tf.abs(log_y_pred))
        self.zero_penalty.assign_add(zero_penalty)
        self.num_batches.assign_add(1.0)

    def result(self):
        return self.zero_penalty / self.num_batches

    def reset_states(self):
        self.zero_penalty.assign(0.0)
        self.num_batches.assign(0.0)

class SpearmanCorrelationPerClass(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='spearman_correlation_per_class', **kwargs):
        super(SpearmanCorrelationPerClass, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.correlation_sums = self.add_weight(name='correlation_sums', shape=(num_classes,), initializer='zeros')
        self.update_counts = self.add_weight(name='update_counts', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_classes):
            y_true_class = tf.cast(y_true[:, i], tf.float32)
            y_pred_class = tf.cast(y_pred[:, i], tf.float32)

            non_zero_indices = tf.where(tf.not_equal(y_true_class, 0))
            y_true_non_zero = tf.gather_nd(y_true_class, non_zero_indices)
            y_pred_non_zero = tf.gather_nd(y_pred_class, non_zero_indices)

            # Use tf.cond for graph-compatible conditional execution
            def compute_correlation():
                order = tf.argsort(y_true_non_zero)
                ranks_true = tf.cast(tf.argsort(order), tf.float32)

                order_pred = tf.argsort(y_pred_non_zero)
                ranks_pred = tf.cast(tf.argsort(order_pred), tf.float32)

                rank_diffs = ranks_true - ranks_pred
                rank_diffs_squared_sum = tf.reduce_sum(tf.square(rank_diffs))
                n = tf.cast(tf.size(y_true_non_zero), tf.float32)

                correlation = 1 - (6 * rank_diffs_squared_sum) / (n * (n*n - 1))
                correlation = tf.where(tf.math.is_nan(correlation), 0.0, correlation)

                return correlation

            def no_correlation():
                return 0.0

            correlation = tf.cond(tf.size(y_true_non_zero) > 1, compute_correlation, no_correlation)

            indices = [[i]]  # Indices to update
            self.correlation_sums.assign(tf.tensor_scatter_nd_add(self.correlation_sums, indices, [correlation]))
            self.update_counts.assign(tf.tensor_scatter_nd_add(self.update_counts, indices, [tf.cast(tf.size(y_true_non_zero) > 1, tf.float32)]))


    def result(self):
        valid_counts = tf.where(self.update_counts > 0, self.update_counts, tf.ones_like(self.update_counts))
        avg_correlations = self.correlation_sums / valid_counts
        return tf.reduce_mean(avg_correlations)

    def reset_states(self):
        self.correlation_sums.assign(tf.zeros_like(self.correlation_sums))
        self.update_counts.assign(tf.zeros_like(self.update_counts))


class PearsonCorrelationLog(tf.keras.metrics.Metric):
    def __init__(self, name='pearson_correlation_log', **kwargs):
        super(PearsonCorrelationLog, self).__init__(name=name, **kwargs)
        self.y_true_sum = self.add_weight(name='y_true_sum', initializer='zeros')
        self.y_pred_sum = self.add_weight(name='y_pred_sum', initializer='zeros')
        self.y_true_squared_sum = self.add_weight(name='y_true_squared_sum', initializer='zeros')
        self.y_pred_squared_sum = self.add_weight(name='y_pred_squared_sum', initializer='zeros')
        self.y_true_y_pred_sum = self.add_weight(name='y_true_y_pred_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.where(y_pred < 0, tf.zeros_like(y_pred), y_pred)

        y_true = tf.math.log(y_true*1000 + 1)
        y_pred = tf.math.log(y_pred*1000 + 1)

        self.y_true_sum.assign_add(tf.reduce_sum(y_true))
        self.y_pred_sum.assign_add(tf.reduce_sum(y_pred))
        self.y_true_squared_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.y_pred_squared_sum.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.y_true_y_pred_sum.assign_add(tf.reduce_sum(y_true * y_pred))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        numerator = self.count * self.y_true_y_pred_sum - self.y_true_sum * self.y_pred_sum
        denominator = tf.sqrt((self.count * self.y_true_squared_sum - tf.square(self.y_true_sum)) * 
                              (self.count * self.y_pred_squared_sum - tf.square(self.y_pred_sum)))

        return numerator / (denominator + tf.keras.backend.epsilon())

    def reset_states(self):
        self.y_true_sum.assign(0.0)
        self.y_pred_sum.assign(0.0)
        self.y_true_squared_sum.assign(0.0)
        self.y_pred_squared_sum.assign(0.0)
        self.y_true_y_pred_sum.assign(0.0)
        self.count.assign(0.0)
