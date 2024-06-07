"""Get a loss based on a chosen task."""

from __future__ import annotations

import tensorflow as tf


def default_loss(task: str) -> tf.keras.losses.Loss:
    """
    Returns a loss based on the specified task.

    If the task is 'topic_classification', returns a BinaryCrossentropy loss. If the task is 'peak_regression', returns

    Parameters
    ----------
    task : str
        The task for which to return a loss. Can be 'topic_classification' or 'peak_regression'.

    Returns
    -------
    tf.keras.losses.Loss
        The default loss function for the given task.
    """
    if task == "topic_classification":
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)
    elif task == "peak_regression":
        return "pass"
    else:
        raise ValueError(f"Unknown task: {task}")
