"""
Model explanation functions using 'gradient x input'-based methods.

Adapted from: https://github.com/p-koo/tfomics/blob/master/tfomics/
"""

from __future__ import annotations

from collections.abc import Callable

import keras
import numpy as np
import torch


def _saliency_map(
        X: torch.Tensor,
        model: keras.Model,
        class_index: int | None = None,
        func: Callable[[torch.Tensor], torch.Tensor] = torch.mean
    ) -> torch.Tensor:
    """Fast function to generate saliency maps."""
    if func is None:
        func = torch.mean
    X = X.clone().detach().requires_grad_(True)
    outputs = model(X)
    if class_index is not None:
        outputs = outputs[:, class_index]
    else:
        outputs = func(outputs)

    outputs.backward(torch.ones_like(outputs))
    return X.grad

def _smoothgrad(
    x: torch.Tensor,
    model: keras.Model,
    num_samples: int = 50,
    mean: float = 0.0,
    stddev: float = 0.1,
    class_index: int | None = None,
    func: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
):
    """Calculate the smoothgrad for a given sequence."""
    _, L, A = x.shape
    x_noise = x.repeat((num_samples, 1, 1)) + torch.normal(
        mean, stddev, size=(num_samples, L, A)
    )
    grad = _saliency_map(x_noise, model, class_index=class_index, func=func)
    return torch.mean(grad, dim=0, keepdim=True).numpy()

def function_batch(
        X: np.ndarray | torch.Tensor,
        fun: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int = 128,
        **kwargs
    ) -> np.ndarray:
    """Run a function in batches.

    Parameters
    ----------
    X
        Sequence inputs, of shape (batch, ...). Can be numpy array or torch tensor.
    fun
        A function that takes a torch.Tensor and returns a torch.Tensor of gradients/importances of the same shape.
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
    if not torch.is_tensor(X):
        X = torch.tensor(X)

    data_size = X.shape[0]
    if data_size <= batch_size:
        return fun(X, **kwargs).detach().cpu().numpy()
    else:
        outputs = np.zeros_like(X)
        n_batches = data_size // batch_size
        for batch_i in range(n_batches):
            batch_start = (batch_i-1)*batch_size
            batch_end = batch_i*batch_size
            outputs[batch_start:batch_end, ...] = fun(X[batch_start:batch_end, ...], **kwargs).detach().cpu().numpy()
        if (n_batches % X.shape[0]) > 0:
            outputs[batch_end:, ...] = fun(X[batch_end: , ...], **kwargs).detach().cpu().numpy()
        return outputs
