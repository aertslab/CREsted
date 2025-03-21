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
    class_index: int | None = None,
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
        Index of model output to explain. Model assumed to return outputs of shape (batch_size, n_classes) if using this.
    func
        Function to reduce model outputs to one value with, if not using class_index.

    Returns
    -------
    Gradients of the same shape as X, (batch, seq_len, nuc).
    """
    if func is None:
        func = tf.math.reduce_mean
    with tf.GradientTape() as tape:
        tape.watch(X)
        if class_index is not None:
            outputs = model(X, training=False)[:, class_index]
        else:
            outputs = func(model(X, training=False))
    return tape.gradient(outputs, X)


@tf.function
def _hessian(X, model, class_index=None, func=tf.math.reduce_mean):
    """Fast function to generate saliency maps."""
    with tf.GradientTape() as t2:
        t2.watch(X)
        with tf.GradientTape() as t1:
            t1.watch(X)
            if class_index is not None:
                outputs = model(X)[:, class_index]
            else:
                outputs = func(model(X))
        g = t1.gradient(outputs, X)
    return t2.jacobian(g, X)


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
