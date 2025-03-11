"""
Model explanation functions using 'gradient x input'-based methods in torch.

Adapted from: https://github.com/p-koo/tfomics/blob/master/tfomics/
"""

import numpy as np
import torch


class Explainer:
    """Wrapper class for attribution maps."""

    def __init__(self, model, class_index=None, func=torch.mean, batch_size=128, seed=None):
        """Initialize the explainer."""
        self.model = model
        self.class_index = class_index
        self.func = func
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def saliency_maps(self, X):
        """Calculate saliency maps for a given sequence."""
        return function_batch(
            X,
            saliency_map,
            self.batch_size,
            model=self.model,
            class_index=self.class_index,
            func=self.func,
        )

    def smoothgrad(self, X, num_samples=50, mean=0.0, stddev=0.1):
        """Calculate smoothgrad for a given sequence."""
        return function_batch(
            X,
            smoothgrad,
            batch_size=1,
            model=self.model,
            num_samples=num_samples,
            mean=mean,
            stddev=stddev,
            class_index=self.class_index,
            func=self.func,
        )

    def integrated_grad(self, X, baseline_type="random", num_steps=25):
        """Calculate integrated gradients for a given sequence."""
        scores = []
        for x in X:
            x = torch.tensor(np.expand_dims(x, axis=0), dtype=torch.float32)
            baseline = torch.tensor(
                self.set_baseline(x, baseline_type, num_samples=1), dtype=torch.float32
            )
            intgrad_scores = integrated_grad(
                x,
                model=self.model,
                baseline=baseline,
                num_steps=num_steps,
                class_index=self.class_index,
                func=self.func,
                batch_size=self.batch_size
            )
            scores.append(intgrad_scores)
        return np.concatenate(scores, axis=0)

    def expected_integrated_grad(
        self, X, num_baseline=25, baseline_type="random", num_steps=25
    ):
        """Calculate expected integrated gradients for a given sequence."""
        scores = []
        for x in X:
            x = torch.tensor(np.expand_dims(x, axis=0), dtype=torch.float32)
            baselines = torch.tensor(
                self.set_baseline(x, baseline_type, num_samples=num_baseline),
                dtype=torch.float32,
            )
            intgrad_scores = expected_integrated_grad(
                x,
                model=self.model,
                baselines=baselines,
                num_steps=num_steps,
                class_index=self.class_index,
                func=self.func,
                batch_size=self.batch_size
            )
            scores.append(intgrad_scores)
        return np.concatenate(scores, axis=0)

    def mutagenesis(self, X):
        """In silico mutagenesis analysis for a given sequence."""
        scores = []
        for x in X:
            x = torch.tensor(np.expand_dims(x, axis=0), dtype=torch.float32)
            scores.append(mutagenesis(x, self.model, self.class_index, batch_size=self.batch_size))
        return np.concatenate(scores, axis=0)

    def set_baseline(self, x, baseline, num_samples):
        """Generate the background sequences."""
        if baseline == "random":
            baseline = self.random_shuffle(x, num_samples)
        else:
            baseline = np.zeros(x.shape, dtype=np.float32)
        return baseline

    def random_shuffle(self, x, num_samples):
        """Randomly shuffle sequences. Assumes x shape is (1, L, A), returns (num_samples, L, A)."""
        _, L, A = x.shape
        x_shuffle = np.zeros((num_samples, L, A), dtype=x.dtype)
        for i in range(num_samples):
            x_shuffle[i, ...] = self.rng.permuted(x, axis = -2)
        return x_shuffle


def saliency_map(X, model, class_index=None, func=torch.mean):
    """Fast function to generate saliency maps."""
    X = X.clone().detach().requires_grad_(True)

    outputs = model(X)
    if class_index is not None:
        outputs = outputs[:, class_index]
    else:
        outputs = func(outputs)

    outputs.backward(torch.ones_like(outputs))
    return X.grad


def smoothgrad(
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
    grad = saliency_map(x_noise, model, class_index=class_index, func=func)
    return torch.mean(grad, dim=0, keepdim=True).numpy()


def integrated_grad(
    x, model, baseline, num_steps=25, class_index=None, func=torch.mean, batch_size=128
):
    """Calculate integrated gradients for a given sequence."""

    def integral_approximation(gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / 2.0
        integrated_gradients = np.mean(grads, axis=0, keepdims=True)
        return integrated_gradients

    steps = torch.linspace(0.0, 1.0, num_steps + 1)
    x_interp = interpolate_data(baseline, x, steps)
    grad = function_batch(
            x_interp,
            saliency_map,
            model=model,
            class_index=class_index,
            func=func,
            batch_size=batch_size,
        )
    avg_grad = integral_approximation(grad)
    return avg_grad


def expected_integrated_grad(
    x, model, baselines, num_steps=25, class_index=None, func=torch.mean, batch_size=128,
):
    """Average integrated gradients across different backgrounds."""
    def integral_approximation(gradients):
        grads = (gradients[:, :-1, ...] + gradients[:, 1:, ...]) / 2.0
        integrated_gradients = np.mean(grads, axis=1)
        return integrated_gradients

    # Make x: for each baseline, shuffle
    x_full = []
    for baseline in baselines:
        steps = torch.linspace(0.0, 1.0, num_steps + 1)
        x_interp = interpolate_data(baseline, x, steps)
        x_full.append(x_interp)
    x_full = torch.concat(x_full, axis=0)
    # Calculate grads
    grad = function_batch(
        x_full,
        saliency_map,
        model=model,
        class_index=class_index,
        func=func,
        batch_size=batch_size,
    )
    # Reshape from n_seqs*n_baselines*n_steps, seq_len, 4 to n_baselines, n_steps, seq_len, 4
    grad = grad.reshape([len(baselines), num_steps+1, x.shape[-2], x.shape[-1]])

    # Apply integrated gradient transform
    avg_grad = integral_approximation(grad)

    # Apply expected gradient transforms
    avg_grad = np.mean(avg_grad, axis=0, keepdims=True)
    return avg_grad

def interpolate_data(baseline, x, steps):
    steps_x = steps[:, None, None]
    delta = x - baseline
    x = baseline + steps_x * delta
    return x


def mutagenesis(x, model, class_index=None, batch_size=None):
    """In silico mutagenesis analysis for a given sequence."""

    def generate_mutagenesis(x):
        _, L, A = x.shape
        x_mut = []
        for length in range(L):
            for a in range(A):
                x_new = np.copy(x)
                x_new[0, length, :] = 0
                x_new[0, length, a] = 1
                x_mut.append(x_new)
        return np.concatenate(x_mut, axis=0)

    def reconstruct_map(predictions):
        _, L, A = x.shape

        mut_score = np.zeros((1, L, A))
        k = 0
        for length in range(L):
            for a in range(A):
                mut_score[0, length, a] = predictions[k]
                k += 1
        return mut_score

    def get_score(x, model, class_index):
        score = model(torch.tensor(x).float()).detach().cpu().numpy()
        if class_index is None:
            score = np.sqrt(np.sum(score**2, axis=-1, keepdims=True))
        else:
            score = score[:, class_index]
        return score

    # generate mutagenized sequences
    x_mut = generate_mutagenesis(x)

    # get baseline wildtype score
    wt_score = get_score(x, model, class_index)
    predictions = get_score(x_mut, model, class_index)

    # reshape mutagenesis predictions
    mut_score = reconstruct_map(predictions)

    return mut_score - wt_score


def grad_times_input(x, scores):
    """Compute the gradient times input."""
    new_scores = []
    for i, score in enumerate(scores):
        new_scores.append(np.sum(x[i] * score, axis=1))
    return np.array(new_scores)


def l2_norm(scores):
    """Compute the L2 norm of the scores."""
    return np.sum(np.sqrt(scores**2), axis=2)


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
