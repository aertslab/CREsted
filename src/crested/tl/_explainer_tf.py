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
        func: Callable[[tf.Tensor], tf.Tensor] = tf.math.reduce_mean
    ) -> tf.Tensor:
    """Fast function to generate saliency maps."""
    if func is None:
        func = tf.math.reduce_mean
    with tf.GradientTape() as tape:
        tape.watch(X)
        if class_index is not None:
            outputs = model(X, training = False)[:, class_index]
        else:
            outputs = func(model(X, training = False))
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
    class_index = None,
    func: Callable[[tf.Tensor], tf.Tensor] = tf.math.reduce_mean,
):
    """Calculate smoothgrad for a given sequence."""
    _, L, A = x.shape
    x_noise = tf.tile(x, (num_samples, 1, 1)) + tf.random.normal(
        (num_samples, L, A), mean, stddev
    )
    grad = _saliency_map(x_noise, model, class_index=class_index, func=func)
    return tf.reduce_mean(grad, axis=0, keepdims=True)

def function_batch(
        X: np.ndarray | tf.Tensor,
        fun: Callable[[tf.Tensor], tf.Tensor],
        batch_size: int = 128,
        **kwargs
    ) -> np.ndarray:
    """Run a function in batches.

    Parameters
    ----------
    X
        Sequence inputs, of shape (batch, ...). Can be numpy array or tf tensor.
    fun
        A function that takes a tf.Tensor and returns a tf.Tensor of gradients/importances of the same shape.
    model
        Your Keras model.
    batch_size
        Batch size to use when calculating gradients with the model.
        Default is 128.
    kwargs
        Passed to fun().

    Returns
    -------
    Numpy array of the same shape as X.
    """
    if not tf.is_tensor(X):
        X = tf.convert_to_tensor(X)

    data_size = X.shape[0]
    if data_size <= batch_size:
        return fun(X, **kwargs).numpy()
    else:
        outputs = np.zeros_like(X)
        n_batches = data_size // batch_size
        for batch_i in range(n_batches):
            batch_start = (batch_i)*batch_size
            batch_end = (batch_i+1)*batch_size
            outputs[batch_start:batch_end, ...] = fun(X[batch_start:batch_end, ...], **kwargs).numpy()
        if (n_batches % X.shape[0]) > 0:
            outputs[batch_end:, ...] = fun(X[batch_end: , ...], **kwargs).numpy()
        return outputs
