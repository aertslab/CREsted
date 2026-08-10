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
    class_index: int | list[int] | None = None,
    func: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> torch.Tensor:
    """Fast function to generate saliency maps.

    Parameters
    ----------
    X
        torch.Tensor of sequences/model inputs, of shape (n_sequences, seq_len, nuc).
    model
        Your Keras model, or any object that supports __call__ with gradients, so it can also be a non-Keras PyTorch model.
    class_index
        Index (or list of indices) of model output(s) to explain. Model assumed to return outputs of shape (batch_size, n_classes) if using this.
        If a list, gradients for all requested classes are computed from a single forward pass, which is reused across classes.
    func
        Function to reduce model outputs to one value with, for any class_index entries that are None.

    Returns
    -------
    Gradients of the same shape as X, (batch, seq_len, nuc), if class_index is a single int or None.
    If class_index is a list, gradients of shape (batch, n_classes, seq_len, nuc).
    """
    if func is None:
        func = torch.mean
    is_multi_class = isinstance(class_index, (list, tuple))
    class_indices = class_index if is_multi_class else [class_index]
    X = X.clone().detach().requires_grad_(True)
    outputs = model(X)
    n = len(class_indices)
    grads = []
    for i, idx in enumerate(class_indices):
        target = outputs[:, idx] if idx is not None else func(outputs)
        grad = torch.autograd.grad(target, X, grad_outputs=torch.ones_like(target), retain_graph=(i < n - 1))[0]
        grads.append(grad)
    return torch.stack(grads, dim=1) if is_multi_class else grads[0]


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
    x_noise = x.repeat((num_samples, 1, 1)) + torch.normal(mean, stddev, size=(num_samples, L, A))
    grad = _saliency_map(x_noise, model, class_index=class_index, func=func)
    return torch.mean(grad, dim=0, keepdim=True)


def _is_tensor(array) -> bool:
    return torch.is_tensor(array)


def _to_tensor(array: np.array) -> torch.Tensor:
    return torch.from_numpy(array)


def _from_tensor(tensor: torch.Tensor) -> np.array:
    return tensor.detach().cpu().numpy()
