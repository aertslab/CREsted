"""
Model explanation functions using 'gradient x input'-based methods.

Adapted from: https://github.com/p-koo/tfomics/blob/master/tfomics/
"""

import numpy as np
import tensorflow as tf

from crested.utils._seq_utils import generate_mutagenesis


class Explainer:
    """wrapper class for attribution maps."""

    def __init__(self, model, class_index=None, func=tf.math.reduce_mean, batch_size=128, seed = None):
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
            batch_size=self.batch_size,
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
        # scores = []
        # for x in X:
        #     x = np.expand_dims(x, axis=0)
        #     baseline = self.set_baseline(x, baseline_type, num_samples=1)
        #     intgrad_scores = integrated_grad(
        #         x,
        #         model=self.model,
        #         baseline=baseline,
        #         num_steps=num_steps,
        #         class_index=self.class_index,
        #         func=self.func,
        #     )
        #     scores.append(intgrad_scores)
        # return np.concatenate(scores, axis=0)
        return integrated_grad(X, self.model, num_steps=num_steps, num_baseline=1, baseline_type=baseline_type, func=self.func, batch_size = self.batch_size, seed = 42)


    # def expected_integrated_grad(
    #     self, X, num_baseline=25, baseline_type="random", num_steps=25
    # ):
    #     """
    #     Average integrated gradients across different backgrounds.

    #     Calculates expected integrated gradients with num_baseline > 1, integrated gradients with num_baseline == 1.
    #     """
    #     scores = []
    #     for x in X:
    #         x = np.expand_dims(x, axis=0)
    #         baselines = self.set_baseline(x, baseline_type, num_samples=num_baseline)
    #         intgrad_scores = integrated_grad(
    #             x,
    #             model=self.model,
    #             baselines=baselines,
    #             num_steps=num_steps,
    #             class_index=self.class_index,
    #             func=self.func,
    #             batch_size=self.batch_size,
    #         )
    #         scores.append(intgrad_scores)
    #     return np.concatenate(scores, axis=0)

    def expected_integrated_grad(
        self, X, num_baseline=25, baseline_type="random", num_steps=25, seed = 42
    ):
        """
        Average integrated gradients across different backgrounds.

        Calculates expected integrated gradients with num_baseline > 1, integrated gradients with num_baseline == 1.
        """
        return integrated_grad(X, self.model, num_steps=num_steps, num_baseline=num_baseline, baseline_type=baseline_type, func=self.func, batch_size = self.batch_size, seed = seed)

    def mutagenesis(self, X):
        """In silico mutagenesis analysis for a given sequence."""
        scores = []
        for x in X:
            x = np.expand_dims(x, axis=0)
            scores.append(mutagenesis(x, self.model, self.class_index, batch_size=self.batch_size))
        return np.concatenate(scores, axis=0)

    def set_baseline(self, x, baseline, num_samples):
        """Set the background for integrated gradients."""
        if baseline == "random":
            baseline = self.random_shuffle(x, num_samples)
        else:
            baseline = np.zeros(x.shape)
        return baseline

    def random_shuffle(self, x, num_samples):
        """Randomly shuffle sequences. Assumes x shape is (1, L, A), returns (num_samples, L, A)."""
        _, L, A = x.shape # Assumes x only has one entry (_ should be 1)
        x_shuffle = np.zeros((num_samples, L, A), dtype=x.dtype)
        for i in range(num_samples):
            x_shuffle[i, ...] = self.rng.permuted(x, axis = -2)
        return x_shuffle

def make_baseline(X, baseline_type, num_samples = None, seed = 42):
    """Set the background for integrated gradients. Assumes x shape is (B, L, A), returns (B, num_samples, L, A) or (B, 1, L, A)."""
    if baseline_type == "random":
        if num_samples is None:
           raise ValueError("If using random baseline, num_samples must be set.")
        return random_shuffle(X, num_samples, seed)
    else:
        # If using zeroes, extra samples is useless since they're all equivalent, so keeping it as 1 sample
        return np.expand_dims(np.zeros_like(X), axis = 1)


def random_shuffle(X, num_samples, seed = 42):
    """Randomly shuffle sequences. Assumes x shape is (B, L, A), returns (B, num_samples, L, A)."""
    rng = np.random.default_rng(seed=seed)
    B, L, A = X.shape
    x_shuffle = np.zeros((B, num_samples, L, A), dtype=X.dtype)
    for seq_i, x in enumerate(X):
        for sample_i in range(num_samples):
            x_shuffle[seq_i, sample_i, ...] = rng.permuted(x, axis = -2)
    return x_shuffle

def saliency_map(X, model, class_index=None, func=tf.math.reduce_mean):
    """Fast function to generate saliency maps."""
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

# Integrated grad that only handles a single sequence and single baseline
# def integrated_grad(
#     x, model, baseline, num_steps=25, class_index=None, func=tf.math.reduce_mean, batch_size=128,
# ):
#     """Calculate integrated gradients for a given sequence."""

#     def integral_approximation(gradients):
#         # riemann_trapezoidal
#         grads = (gradients[:-1] + gradients[1:]) / 2.0
#         integrated_gradients = np.mean(grads, axis=0, keepdims=True)
#         return integrated_gradients

#     steps = tf.linspace(start=0.0, stop=1.0, num=num_steps + 1)
#     x_interp = interpolate_data(baseline, x, steps)
#     grad = function_batch(
#             x_interp,
#             saliency_map,
#             model=model,
#             class_index=class_index,
#             func=func,
#             batch_size=batch_size,
#         )
#     avg_grad = integral_approximation(grad)
#     return avg_grad


# Integrated grad that only handles a single sequence
# def integrated_grad(
#     x, model, baselines, num_steps=25, class_index=None, func=tf.math.reduce_mean, batch_size=128
# ):
#     """Average integrated gradients across different backgrounds."""
#     def integral_approximation(gradients):
#         grads = (gradients[:, :-1, ...] + gradients[:, 1:, ...]) / 2.0
#         integrated_gradients = np.mean(grads, axis=1)
#         return integrated_gradients

#     # Make x: for each baseline, shuffle
#     x_full = []
#     for baseline in baselines:
#         steps = tf.linspace(start=0.0, stop=1.0, num=num_steps + 1)
#         x_interp = interpolate_data(baseline, x, steps)
#         x_full.append(x_interp)
#     x_full = tf.concat(x_full, axis=0)

#     # Calculate grads
#     grad = function_batch(
#         x_full,
#         saliency_map,
#         model=model,
#         class_index=class_index,
#         func=func,
#         batch_size=batch_size,
#     )
#     # Reshape from n_baselines*n_steps, seq_len, 4 to n_baselines, n_steps, seq_len, 4
#     grad = grad.reshape([len(baselines), num_steps+1, x.shape[-2], x.shape[-1]])

#     # Apply integrated gradient transform
#     avg_grad = integral_approximation(grad)

#     # Apply expected gradient transforms
#     avg_grad = np.mean(avg_grad, axis=0, keepdims=True)
#     return avg_grad

def integrated_grad(
    X, model, class_index=None, num_baseline=25, num_steps=25, baseline_type = "random", func=tf.math.reduce_mean, batch_size=128, seed = 42,
):
    """Average integrated gradients across different backgrounds."""

    def interpolate_data(x, baseline, steps):
        """
        Interpolate len(steps) sequences from baseline to x.

        Parameters
        ----------
        x
            Main sequence, of shape (1, L, A)
        baseline
            Baseline sequence to interpolate from, of shape (1, L, A)
        steps
            Steps to interpolate at, as 1d vector of [0, 1] values, ideally including both 0 and 1 to include baseline and x in result.
            Easiest approach is tf.linspace(0.0, 1.0, num_steps).
        """
        # steps_x.shape = (n_steps, 1, 1)
        # delta: x/baseline shape
        # final x shape: n_steps, L, A)
        steps_x = steps[:, tf.newaxis, tf.newaxis]
        delta = x - baseline
        x = baseline + steps_x * delta
        return x

    def integral_approximation(gradients):
        grads = (gradients[..., :-1, :, :] + gradients[..., 1:, :, :]) / 2.0
        integrated_gradients = np.mean(grads, axis=-3)
        return integrated_gradients

    # If one seq, expand:
    if X.ndim == 2:
        X = np.expand_dims(X, 0)
    B, L, A = X.shape

    # # Make baselines (shuffled sequences), and for each baseline, interpolate from baseline to x
    X_full = np.zeros((B*num_baseline*(num_steps+1), L, A))
    steps = tf.linspace(start=0.0, stop=1.0, num=num_steps + 1)
    baselines = make_baseline(X, num_samples = num_baseline, baseline_type = baseline_type)
    for batch_idx in range(B):
        for baseline_idx in range(num_baseline):
            start_idx = batch_idx*num_baseline+baseline_idx*(num_steps+1)
            stop_idx = start_idx + num_steps+1
            X_full[start_idx:stop_idx, :, :] = interpolate_data(x = X[batch_idx, :, :], baseline = baselines[batch_idx, baseline_idx, :, :], steps = steps)

    # Calculate grads
    grad = function_batch(
        X_full,
        saliency_map,
        model=model,
        class_index=class_index,
        func=func,
        batch_size=batch_size,
    )
    # Reshape from n_seqs*num_baseline*num_steps, seq_len, 4 to n_seqs, n_baselines, num_steps, seq_len, 4
    grad = grad.reshape([B, num_baseline, num_steps+1, L, A])

    # Apply integrated gradient transform (reduce num_steps down to 1)
    # (n_seqs, n_baselines, num_steps, seq_len, 4) to (n_seqs, n_baselines, seq_len, 4)
    grad = integral_approximation(grad)

    # Apply expected transform (reduce n_baselines down to 1)
    # (n_seqs, n_baselines, seq_len, 4) to  (n_seqs, seq_len, 4)
    grad = np.mean(grad, axis=-3)

    return grad


def mutagenesis(x, model, class_index=None, batch_size=None):
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

    def get_score(x, model, class_index, batch_size=None):
        score = model.predict(x, verbose=0, batch_size=batch_size)
        if class_index is None:
            score = np.sqrt(np.sum(score**2, axis=-1, keepdims=True))
        else:
            score = score[:, class_index]
        return score

    # generate mutagenized sequences
    x_mut = generate_mutagenesis(x)

    # get baseline wildtype score
    wt_score = get_score(x, model, class_index, batch_size=batch_size)
    predictions = get_score(x_mut, model, class_index, batch_size=batch_size)

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
