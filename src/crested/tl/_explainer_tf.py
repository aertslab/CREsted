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
        low_gpu: bool = False,
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
    low_gpu
        Move each batch to/from CPU separately, instead of whole input and output array. Saves GPU memory, but reduces speed.
    kwargs
        Passed to fun().

    Returns
    -------
    Numpy array of the same shape as X.
    """
    # If low_gpu: convert to/from numpy+cpu at batch level for inputs & outputs
    # Else: convert to/from numpy+cpu at X level for inputs & outputs

    if not low_gpu and not tf.is_tensor(X):
        X = tf.convert_to_tensor(X)

    data_size = X.shape[0]
    # If fits in one batch, return directly
    if data_size <= batch_size:
        return fun(X, **kwargs).numpy()
    # Else, loop for as many batches as needed
    else:
        # Save outputs to numpy array (if low_gpu) or tf.Variable (if not low_gpu)
        outputs = np.zeros_like(X) if low_gpu else tf.Variable(tf.zeros_like(X))

        # Get batch indices
        n_batches = data_size // batch_size
        batch_idxes = [(i*batch_size, (i+1)*batch_size) for i in range(n_batches)]
        if data_size % batch_size > 0:
            batch_idxes.append((batch_idxes[-1][1], data_size))

        # Loop over batches
        for batch_start, batch_end in batch_idxes:
            # Get inputs
            batch = X[batch_start:batch_end, ...]
            if low_gpu and not tf.is_tensor(batch):
                batch = tf.convert_to_tensor(batch)
            # Calculate gradients for batch
            grads = fun(batch, **kwargs)
            # Save gradients
            outputs[batch_start:batch_end, ...] = grads.numpy() if low_gpu else grads

        # Return outputs, converting to CPU if not low_gpu
        return outputs if low_gpu else outputs.numpy()
