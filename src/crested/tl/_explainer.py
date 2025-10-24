"""
Model explanation functions using 'gradient x input'-based or mutagenesis-based methods.

Loosely adapted from: https://github.com/p-koo/tfomics/blob/master/tfomics/
"""
from __future__ import annotations

import os
from collections.abc import Callable

import keras
import numpy as np

from crested.utils._seq_utils import generate_mutagenesis, generate_window_shuffle

if os.environ["KERAS_BACKEND"] == "tensorflow":
    from tensorflow import Tensor

    from crested.tl._explainer_tf import (
        _from_tensor,
        _is_tensor,
        _saliency_map,
        _smoothgrad,
        _to_tensor,
    )
elif os.environ["KERAS_BACKEND"] == "torch":
    from torch import Tensor

    from crested.tl._explainer_torch import (
        _from_tensor,
        _is_tensor,
        _saliency_map,
        _smoothgrad,
        _to_tensor,
    )


# ---- Explainer functions ----
def saliency_map(
    X: np.ndarray,
    model: keras.Model,
    class_index: int | None,
    batch_size: int = 128,
    func: Callable = None,
) -> np.ndarray:
    """Calculate saliency maps for a given (set of) sequence(s).

    Parameters
    ----------
    X
        Sequence inputs, of shape (batch, seq_len, 4). Can be numpy array or tf/torch tensor.
    model
        Your Keras model.
    class_index
        The index of the class to explain. If None, applies func to average/sum/etc over all classes.
    func
        If class_index is None, how to combine the final predictions (sum/mean/etc). Should work on tensors of your backend.
    batch_size
        Batch size used for gradient calculations. Note that integrated_grad() calculates gradients for background sequences for each main sequence provided,
        so explaining 1 sequence still requires gradients of e.g. 650 sequences (num_baselines*num_steps+1).
        Default is 128, which works well for 2kb input size models but might struggle on bigger models.
    """
    return function_batch(
        X,
        _saliency_map,
        batch_size=batch_size,
        model=model,
        class_index=class_index,
        func=func,
    )


def integrated_grad(
    X: np.ndarray,
    model: keras.Model,
    class_index: int | None = None,
    baseline_type: str = "random",
    num_baselines: int = 25,
    num_steps: int = 25,
    func: Callable = None,
    batch_size: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Average integrated gradients across different backgrounds.

    Calculate expected integrated gradients with baseline_type='random' and num_baselines=25, the default settings.
    To get integrated gradients, use baseline_type="zeros" (which automatically ignores num_baselines).

    Parameters
    ----------
    X
        Sequence inputs, of shape (batch, seq_len, 4). Can be numpy array or tf/torch tensor.
    model
        Your Keras model.
    class_index
        The index of the class to explain. If None, applies func to average/sum/etc over all classes.
    baseline_type
        How to get the baseline sequence to compare your sequence to.
        "random" shuffles each input sequence `num_baselines` times and interpolates from those shuffled versions to the original, as used in expected integrated gradients.
        "zeros" simply creates a single baseline of zeros (ignoring `num_baselines`). This is used for standard integrated gradients.
    num_baselines
        If using baseline_type="random", how many shuffled sequences you want to use as 'background comparison'.
    num_steps
        How many steps to integrate over, from your baseline as starting point, going up to the actual sequence.
    func
        If class_index is None, how to combine the final predictions (sum/mean/etc). Should work on tensors of your backend.
        Default (None) uses tf.math.reduce_mean/torch.mean.
    batch_size
        Batch size used for gradient calculations. Note that integrated_grad() calculates gradients for background sequences for each main sequence provided,
        so explaining 1 sequence still requires gradients of e.g. 650 sequences (num_baselines*num_steps+1).
        Default is 128, which works well for 2kb input size models but might struggle on bigger models.
    seed
        Seed to use for shuffling sequences when using baseline_type "random".
    """

    def interpolate_data(
        x: np.ndarray, baseline: np.ndarray, steps: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate len(steps) sequences from baseline to x.

        Parameters
        ----------
        x
            Main sequence, of shape (1, seq_len, nuc)
        baseline
            Baseline sequence to interpolate from, of shape (1, seq_len, nuc)
        steps
            Steps to interpolate at, as 1d vector of [0, 1] values, ideally including both 0 and 1 to include baseline and x in result.
            Easiest approach is np.linspace(0.0, 1.0, num_steps).

        Returns
        -------
        Interpolated sequences of shape (n_steps, seq_len, nuc).
        """
        # steps_x.shape = (n_steps, 1, 1)
        # delta: (1, seq_len, nuc)
        # final x shape: (n_steps, seq_len, nuc)
        steps_x = steps[:, None, None]
        delta = x - baseline
        x = baseline + steps_x * delta
        return x

    def integral_approximation(gradients: np.ndarray) -> np.ndarray:
        """
        Calculate approximated integral over your sequence, the baseline, and all intermediate steps.

        Parameters
        ----------
        gradients
            gradients, of shape (batch, n_steps, seq_len, nuc).

        Returns
        -------
        Integrated gradients of shape (batch, seq_len, nuc).
        """
        grads = (gradients[:, :-1, ...] + gradients[:, 1:, ...]) / 2.0
        return np.mean(grads, axis=1)

    # Make baselines
    baselines = make_baselines(
        X, num_samples=num_baselines, baseline_type=baseline_type, seed=seed
    )

    outputs = np.zeros_like(X)
    for i, x in enumerate(X):
        x = np.expand_dims(x, axis=0)

        # Make x: for each baseline, interpolate from baseline to sequence
        x_full = []
        for baseline in baselines[i, ...]:
            steps = np.linspace(start=0.0, stop=1.0, num=num_steps + 1)
            x_interp = interpolate_data(baseline, x, steps)
            x_full.append(x_interp)
        x_full = np.concatenate(x_full, axis=0)

        # Calculate grads
        grad = function_batch(
            x_full,
            _saliency_map,
            model=model,
            class_index=class_index,
            func=func,
            batch_size=batch_size,
        )
        # Reshape from n_baselines*n_steps, seq_len, 4 to n_baselines, n_steps, seq_len, 4
        grad = grad.reshape([num_baselines, num_steps + 1, x.shape[-2], x.shape[-1]])

        # Apply integrated gradient transform
        avg_grad = integral_approximation(grad)

        # Apply expected gradient transforms
        outputs[i, ...] = np.mean(avg_grad, axis=0)
    return outputs


def mutagenesis(
    X: np.ndarray, model: keras.Model, class_index: int = None, batch_size: int = 256
) -> np.ndarray:
    """In silico mutagenesis analysis for a given sequence.

    Parameters
    ----------
    X
        Sequence inputs, of shape (batch, seq_len, 4). Can be numpy array or tf/torch tensor.
    model
        Your Keras model.
    class_index
        The index of the class to explain.
    batch_size
        Batch size to use when predicting values with the model. Note that mutagenesis requires (seq_len*3+1) predictions to explain one sequence.
        Default is 256.
    """

    def reconstruct_map(predictions):
        _, L, A = x.shape

        mut_score = np.zeros((1, L, A))
        k = 0
        for length in range(L):
            for a in range(A):
                mut_score[0, length, a] = predictions[k]
                k += 1
        return mut_score

    def get_score(x, model, class_index, batch_size=None):
        score = model.predict(x, verbose=0, batch_size=batch_size)
        if class_index is None:
            score = np.sqrt(np.sum(score**2, axis=-1, keepdims=True))
        else:
            score = score[:, class_index]
        return score

    scores = []
    for x in X:
        x = np.expand_dims(x, axis=0)

        # generate mutagenized sequences
        x_mut = generate_mutagenesis(x)

        # get baseline wildtype score
        wt_score = get_score(x, model, class_index, batch_size=batch_size)
        predictions = get_score(x_mut, model, class_index, batch_size=batch_size)

        # reshape mutagenesis predictions
        mut_score = reconstruct_map(predictions)
        scores.append(mut_score - wt_score)

    return np.concatenate(scores, axis=0)


def window_shuffle(
    X: np.ndarray,
    model: keras.Model,
    class_index: int = None,
    window_size: int = 5,
    n_shuffles: int = 5,
    uniform: bool = False,
    batch_size: int = 256,
) -> np.ndarray:
    """In silico mutagenesis analysis for a given sequence.

    Parameters
    ----------
    X
        Sequence inputs, of shape (batch, seq_len, 4). Can be numpy array or tf/torch tensor.
    model
        Your Keras model.
    class_index
        The index of the class to explain.
    window_size
        Window size to use to shuffle
    n_shuffles
        Number of shuffles
    uniform
        Whether to reshuffle local sequence or replace with uniformly random sequence
    batch_size
        Batch size to use when predicting values with the model. Note that mutagenesis requires (seq_len*3+1) predictions to explain one sequence.
        Default is 256.
    """

    def reconstruct_map(predictions, window_size, n_shuffles):
        _, L, A = x.shape

        mut_score = np.zeros((1, L, A))
        n_mut_per_shuffle = len(predictions) // n_shuffles
        for location in range(L):
            # determine which predictions affect this location
            number_of_changes = np.min([location + 1, window_size, L - location])
            start = np.max([location - window_size + 1, 0])
            indexes = []
            for shuffle in range(n_shuffles):
                offset = shuffle * n_mut_per_shuffle
                indexes.extend(
                    range(start + offset, (start + number_of_changes) + offset)
                )
            mut_score[0, location, :] = np.mean(predictions[indexes])
        return mut_score

    def get_score(x, model, class_index, batch_size=None):
        score = model.predict(x, verbose=0, batch_size=batch_size)
        if class_index is None:
            score = np.sqrt(np.sum(score**2, axis=-1, keepdims=True))
        else:
            score = score[:, class_index]
        return score

    scores = []
    for x in X:
        x = np.expand_dims(x, axis=0)

        # generate mutagenized sequences
        x_mut = generate_window_shuffle(
            x, window_size=window_size, n_shuffles=n_shuffles, uniform=uniform
        )

        # get baseline wildtype score
        wt_score = get_score(x, model, class_index, batch_size=batch_size)
        predictions = get_score(x_mut, model, class_index, batch_size=batch_size)

        # reshape mutagenesis predictions
        mut_score = reconstruct_map(
            predictions, window_size=window_size, n_shuffles=n_shuffles
        )
        scores.append(wt_score - mut_score)
    return np.concatenate(scores, axis=0)


def smoothgrad(
    X: np.ndarray,
    model: keras.Model,
    class_index: int | None,
    num_samples: int = 50,
    mean: float = 0.0,
    stddev: float = 0.1,
    func: Callable = None,
) -> np.ndarray:
    """Calculate smoothgrad for a given (set of) sequence(s)."""
    return function_batch(
        X,
        _smoothgrad,
        batch_size=1,
        model=model,
        num_samples=num_samples,
        mean=mean,
        stddev=stddev,
        class_index=class_index,
        func=func,
    )


# ---- Helper functions ----
def make_baselines(
    X: np.ndarray, baseline_type: str, num_samples: int = 25, seed: int = 42
) -> np.ndarray:
    """Create backgrounds for integrated gradients.

    Assumes x shape is (batch, seq_len, nuc), returns (batch, num_samples, seq_len, nuc) or (batch, 1, seq_len, nuc).

    Parameters
    ----------
    X
        array of sequences, of shape (n_sequences, seq_len, nuc).
    baseline_type
        How to generate the baselines for each sequence.
        "random" shuffles sequences along seq_len.
        "zeros"/"zeroes" provides an array of zeros.
    num_samples
        If using baseline_type "random", how many shuffled sequences to generate. Default is 25.
    seed
        If using baseline_type "random", which seed to set. Default is 42.

    Returns
    -------
    Baselines of shape (batch, num_samples, seq_len, nuc) if using baseline_type="random" or (batch, 1, seq_len, nuc) if using baseline_type="zeros".
    """
    if baseline_type == "random":
        if num_samples is None:
            raise ValueError("If using random baseline, num_samples must be set.")
        return random_shuffle(X, num_samples, seed)
    elif baseline_type == "zeros" or baseline_type == "zeroes":
        # If using zeroes, extra samples is useless since they're all equivalent, so keeping it as 1 sample
        return np.expand_dims(np.zeros_like(X), axis=1)
    else:
        raise ValueError(
            f"Unrecognised baseline_type {baseline_type}. Must be 'random' or 'zeros'/'zeroes'."
        )


def random_shuffle(X: np.ndarray, num_samples: int, seed: int = 42) -> np.ndarray:
    """Randomly shuffle sequences.

    Parameters
    ----------
    X
        array of sequences, of shape (n_sequences, seq_len, nuc).
    num_samples
        How many shuffled sequences to generate.
    seed
        Seed for shuffling. Default is 42.

    Returns
    -------
    Returns shuffled sequences of shape (batch, num_samples, seq_len, nuc).
    """
    B, L, A = X.shape
    x_shuffle = np.zeros((B, num_samples, L, A), dtype=X.dtype)
    for seq_i, x in enumerate(X):
        rng = np.random.default_rng(seed=seed)
        for sample_i in range(num_samples):
            shuffle = rng.permutation(x.shape[-2])
            x_shuffle[seq_i, sample_i, :, :] = x[shuffle, :]
    return x_shuffle


def function_batch(
    X: np.ndarray | Tensor,
    fun: Callable[[Tensor], Tensor],
    batch_size: int = 128,
    **kwargs,
) -> np.ndarray:
    """Run a function in batches.

    Parameters
    ----------
    X
        Sequence inputs, of shape (batch, ...). Can be numpy array or tf/torch tensor.
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
    data_size = X.shape[0]
    # If fits in one batch, return directly
    if data_size <= batch_size:
        if not _is_tensor(X):
            X = _to_tensor(X)
        grads = fun(X, **kwargs)
        return _from_tensor(grads)
    # Else, loop for as many batches as needed
    else:
        outputs = []
        # Get batch indices
        n_batches = data_size // batch_size
        batch_idxes = [(i * batch_size, (i + 1) * batch_size) for i in range(n_batches)]
        if data_size % batch_size > 0:
            batch_idxes.append((batch_idxes[-1][1], data_size))

        # Loop over batches
        for batch_start, batch_end in batch_idxes:
            # Get inputs
            batch = X[batch_start:batch_end, ...]
            if not _is_tensor(batch):
                batch = _to_tensor(batch)
            # Calculate gradients for batch
            grads = fun(batch, **kwargs)
            # Save gradients
            outputs.append(_from_tensor(grads))

        # Return outputs
        return np.concatenate(outputs, axis=0)
