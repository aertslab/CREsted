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
        Index of model output to explain. Model assumed to return outputs of shape (batch_size, n_classes) if using this.
    func
        Function to reduce model outputs to one value with, if not using class_index.

    Returns
    -------
    Gradients of the same shape as X, (batch, seq_len, nuc).
    """
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


def _is_tensor(array) -> bool:
    return torch.is_tensor(array)


def _to_tensor(array: np.array) -> torch.Tensor:
    return torch.from_numpy(array)


def _from_tensor(tensor: torch.Tensor) -> np.array:
    return tensor.detach().cpu().numpy()
