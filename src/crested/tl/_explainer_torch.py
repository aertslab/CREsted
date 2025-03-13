"""
Model explanation functions using 'gradient x input'-based methods.

Adapted from: https://github.com/p-koo/tfomics/blob/master/tfomics/
"""

import numpy as np
import torch


def _saliency_map(X, model, class_index=None, func=torch.mean):
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
    x,
    model,
    num_samples=50,
    mean=0.0,
    stddev=0.1,
    class_index=None,
    func=torch.mean,
):
    """Calculate the smoothgrad for a given sequence."""
    _, L, A = x.shape
    x = torch.tensor(x)
    x_noise = x.repeat((num_samples, 1, 1)) + torch.normal(
        mean, stddev, size=(num_samples, L, A)
    )
    grad = _saliency_map(x_noise, model, class_index=class_index, func=func)
    return torch.mean(grad, dim=0, keepdim=True).numpy()

def function_batch(X, fun, batch_size=128, **kwargs):
    """Run a function in batches."""
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
