"""
Model explanation functions using 'gradient x input'-based methods.

Adapted from: https://github.com/p-koo/tfomics/blob/master/tfomics/
"""

import numpy as np
import tensorflow as tf

### Primitive functions -----

def _saliency_map(X, model, class_index=None, func=tf.math.reduce_mean):
    """Fast function to generate saliency maps."""
    if func is None:
        func = tf.math.reduce_mean

    if not tf.is_tensor(X):
        X = tf.Variable(X)

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
    if not tf.is_tensor(X):
        X = tf.Variable(X)

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
    x,
    model,
    num_samples=50,
    mean=0.0,
    stddev=0.1,
    class_index=None,
    func=tf.math.reduce_mean,
):
    """Calculate smoothgrad for a given sequence."""
    if not tf.is_tensor(x):
        x = tf.Variable(x)
    _, L, A = x.shape
    x_noise = tf.tile(x, (num_samples, 1, 1)) + tf.random.normal(
        (num_samples, L, A), mean, stddev
    )
    grad = _saliency_map(x_noise, model, class_index=class_index, func=func)
    return tf.reduce_mean(grad, axis=0, keepdims=True)

def function_batch(X, fun, batch_size=128, **kwargs):
    """Run a function in batches."""
    if not tf.is_tensor(X):
        X = tf.Variable(X)

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
