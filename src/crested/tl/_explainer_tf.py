"""
Model explanation functions using 'gradient x input'-based methods.

Adapted from: https://github.com/p-koo/tfomics/blob/master/tfomics/
"""

from __future__ import annotations

from collections.abc import Callable

import keras
import numpy as np
import tensorflow as tf


def _saliency_map(
    X: tf.Tensor,
    model: keras.Model,
    class_index: int | list[int] | None = None,
    func: Callable[[tf.Tensor], tf.Tensor] = tf.math.reduce_mean,
) -> tf.Tensor:
    """Fast function to generate saliency maps.

    Parameters
    ----------
    X
        tf.Tensor of sequences/model inputs, of shape (n_sequences, seq_len, nuc).
    model
        Your Keras model, or any object that supports __call__ with gradients, so it can also be a non-Keras TensorFlow model.
    class_index
        Index (or list of indices) of model output(s) to explain. Model assumed to return outputs of shape (batch_size, n_classes) if using this.
    func
        Function to reduce model outputs to one value with, for any class_index entries that are None.

    Returns
    -------
    Gradients of the same shape as X, (batch, seq_len, nuc), if class_index is a single int or None.
    If class_index is a list, gradients of shape (batch, n_classes, seq_len, nuc).
    """
    if func is None:
        func = tf.math.reduce_mean
    is_multi_class = isinstance(class_index, (list, tuple))
    class_indices = class_index if is_multi_class else [class_index]
    with tf.GradientTape(persistent=is_multi_class) as tape:
        tape.watch(X)
        raw_outputs = model(X, training=False)
        targets = [raw_outputs[:, idx] if idx is not None else func(raw_outputs) for idx in class_indices]
    grads = [tape.gradient(target, X) for target in targets]
    if is_multi_class:
        del tape  # release the persistent tape's resources
    return tf.stack(grads, axis=1) if is_multi_class else grads[0]

def _smoothgrad(
    x: tf.Tensor,
    model: keras.Model,
    num_samples: int = 50,
    mean: float = 0.0,
    stddev: float = 0.1,
    class_index=None,
    func: Callable[[tf.Tensor], tf.Tensor] = tf.math.reduce_mean,
):
    """Calculate smoothgrad for a given sequence."""
    _, L, A = x.shape
    x_noise = tf.tile(x, (num_samples, 1, 1)) + tf.random.normal(
        (num_samples, L, A), mean, stddev
    )
    grad = _saliency_map(x_noise, model, class_index=class_index, func=func)
    return tf.reduce_mean(grad, axis=0, keepdims=True)


def _is_tensor(array) -> bool:
    return tf.is_tensor(array)


def _to_tensor(array: np.array) -> tf.Tensor:
    return tf.convert_to_tensor(array)


def _from_tensor(tensor: tf.Tensor) -> np.array:
    return tensor.numpy()
