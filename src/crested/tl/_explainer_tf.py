"""
Model explanation functions using 'gradient x input'-based methods.

Adapted from: https://github.com/p-koo/tfomics/blob/master/tfomics/
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from crested.utils._seq_utils import generate_mutagenesis


class Explainer:
    """wrapper class for attribution maps."""

    def __init__(self, model, class_index=None, func=tf.math.reduce_mean):
        """Initialize the explainer."""
        self.model = model
        self.class_index = class_index
        self.func = func

    def saliency_maps(self, X, batch_size=128):
        """Calculate saliency maps for a given sequence."""
        return function_batch(
            X,
            saliency_map,
            batch_size,
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
            x = np.expand_dims(x, axis=0)
            baseline = self.set_baseline(x, baseline_type, num_samples=1)
            intgrad_scores = integrated_grad(
                x,
                model=self.model,
                baseline=baseline,
                num_steps=num_steps,
                class_index=self.class_index,
                func=self.func,
            )
            scores.append(intgrad_scores)
        return np.array(scores)

    def expected_integrated_grad(
        self, X, num_baseline=25, baseline_type="random", num_steps=25
    ):
        """Average integrated gradients across different backgrounds."""
        scores = []
        for x in X:
            x = np.expand_dims(x, axis=0)
            baselines = self.set_baseline(x, baseline_type, num_samples=num_baseline)
            intgrad_scores = expected_integrated_grad(
                x,
                model=self.model,
                baselines=baselines,
                num_steps=num_steps,
                class_index=self.class_index,
                func=self.func,
            )
            scores.append(intgrad_scores)
        return np.array(scores)

    def mutagenesis(self, X, class_index=None):
        """In silico mutagenesis analysis for a given sequence."""
        scores = []
        for x in X:
            x = np.expand_dims(x, axis=0)
            scores.append(mutagenesis(x, self.model, class_index))
        return np.concatenate(scores, axis=0)

    def set_baseline(self, x, baseline, num_samples):
        """Set the background for integrated gradients."""
        if baseline == "random":
            baseline = random_shuffle(x, num_samples)
        else:
            baseline = np.zeros(x.shape)
        return baseline


def saliency_map(
    X, model, class_index: int | list[int] | None = None, func=tf.math.reduce_mean
):
    """Fast function to generate saliency maps."""
    if not tf.is_tensor(X):
        X = tf.Variable(X)

    # use persistent tape so gradient can be calculated for each output in class_index, in case
    # class_index is a list of indexes.
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        output = model(X)
        if isinstance(class_index, int):
            # get output for class (C)
            outputs_C = [output[:, class_index]]
        elif isinstance(class_index, list):
            # get output for multiple classes
            outputs_C = [output[:, c] for c in class_index]
        elif class_index is None:
            # legacy mode -- not sure if `func` is even needed here??
            outputs_C = [func(model(X))]
        else:
            raise ValueError(
                f"class_index should be either an integer a list of integers or None, not: {class_index}."
            )
    grads = np.empty((len(outputs_C), *X.shape))
    for i in range(len(outputs_C)):
        grads[i] = tape.gradient(outputs_C[i], X)
    # explicitly delete the tape, needed because persistent is True
    del tape
    # squeeze grads so first dimension is dropped in case class_index is a single int.
    return grads.squeeze()


@tf.function
def hessian(X, model, class_index=None, func=tf.math.reduce_mean):
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


def smoothgrad(
    x,
    model,
    num_samples=50,
    mean=0.0,
    stddev=0.1,
    class_index=None,
    func=tf.math.reduce_mean,
):
    """Calculate smoothgrad for a given sequence."""
    _, L, A = x.shape
    x_noise = tf.tile(x, (num_samples, 1, 1)) + tf.random.normal(
        (num_samples, L, A), mean, stddev
    )
    grad = saliency_map(x_noise, model, class_index=class_index, func=func)
    return tf.reduce_mean(grad, axis=0, keepdims=True)


def integrated_grad(
    x,
    model,
    baseline,
    num_steps=25,
    class_index: int | list[int] | None = None,
    func=tf.math.reduce_mean,
):
    """Calculate integrated gradients for a given sequence."""

    def integral_approximation(gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    def interpolate_data(baseline, x, steps):
        steps_x = steps[:, tf.newaxis, tf.newaxis]
        delta = x - baseline
        x = baseline + steps_x * delta
        return x

    steps = tf.linspace(start=0.0, stop=1.0, num=num_steps + 1)
    x_interp = interpolate_data(baseline, x, steps)
    grad = saliency_map(x_interp, model, class_index=class_index, func=func)
    # at this point the shape of grad is either:
    # - (num_steps + 1, *x.shape)                       in case class_index is None or a single int.
    # - (len(class_index), num_steps + 1, *x.shape)     in case class_idnex is a list of int.
    if len(grad.shape) == 4:
        # second case, put num_steps + 1 on first axis
        grad = grad.swapaxes(0, 1)
    avg_grad = integral_approximation(grad)
    if len(avg_grad.shape) != 3:
        # first case, in this case the dimension should be expanded.
        # in the second case they are already expanded.
        avg_grad = np.expand_dims(avg_grad, axis=0)
    return avg_grad


def expected_integrated_grad(
    x, model, baselines, num_steps=25, class_index=None, func=tf.math.reduce_mean
):
    """Average integrated gradients across different backgrounds."""
    grads = []
    for baseline in baselines:
        grads.append(
            integrated_grad(
                x,
                model,
                baseline,
                num_steps=num_steps,
                class_index=class_index,
                func=func,
            )
        )
    return np.mean(np.array(grads), axis=0)


def mutagenesis(x, model, class_index=None):
    """In silico mutagenesis analysis for a given sequence."""

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
        score = model.predict(x, verbose=0)
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
    dataset = tf.data.Dataset.from_tensor_slices(X)
    outputs = []
    for x in dataset.batch(batch_size):
        outputs.append(fun(x, **kwargs))
    return np.concatenate(outputs, axis=0)


def random_shuffle(x, num_samples=1):
    """Randomly shuffle sequences. Assumes x shape is (N,L,A)."""
    x_shuffle = []
    for _ in range(num_samples):
        shuffle = np.random.permutation(x.shape[1])
        x_shuffle.append(x[0, shuffle, :])
    return np.array(x_shuffle)
