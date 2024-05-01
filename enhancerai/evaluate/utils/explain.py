"""
Model explanation functions using 'gradient x input'-based methods.
Taken from: https://github.com/p-koo/tfomics/blob/master/tfomics/
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logomaker


class Explainer:
    """wrapper class for attribution maps"""

    def __init__(self, model, class_index=None, func=tf.math.reduce_mean):
        self.model = model
        self.class_index = class_index
        self.func = func

    def saliency_maps(self, X, batch_size=128):
        return function_batch(
            X,
            saliency_map,
            batch_size,
            model=self.model,
            class_index=self.class_index,
            func=self.func,
        )

    def smoothgrad(self, X, num_samples=50, mean=0.0, stddev=0.1):
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
        return np.concatenate(scores, axis=0)

    def expected_integrated_grad(
        self, X, num_baseline=25, baseline_type="random", num_steps=25
    ):
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
        return np.concatenate(scores, axis=0)

    def mutagenesis(self, X, class_index=None):
        scores = []
        for x in X:
            x = np.expand_dims(x, axis=0)
            scores.append(mutagenesis(x, self.model, class_index))
        return np.concatenate(scores, axis=0)

    def set_baseline(self, x, baseline, num_samples):
        if baseline == "random":
            baseline = random_shuffle(x, num_samples)
        else:
            baseline = np.zeros((x.shape))
        return baseline


# ------------------------------------------------------------------------------
# Fast functions to calculate gradients and hessians


def saliency_map(X, model, class_index=None, func=tf.math.reduce_mean):
    """fast function to generate saliency maps"""
    if not tf.is_tensor(X):
        X = tf.Variable(X)

    with tf.GradientTape() as tape:
        tape.watch(X)
        if class_index is not None:
            outputs = model(X)[:, class_index]
        else:
            outputs = func(model(X))
    return tape.gradient(outputs, X)


@tf.function
def hessian(X, model, class_index=None, func=tf.math.reduce_mean):
    """fast function to generate saliency maps"""
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


# ------------------------------------------------------------------------------


def smoothgrad(
    x,
    model,
    num_samples=50,
    mean=0.0,
    stddev=0.1,
    class_index=None,
    func=tf.math.reduce_mean,
):
    _, L, A = x.shape
    x_noise = tf.tile(x, (num_samples, 1, 1)) + tf.random.normal(
        (num_samples, L, A), mean, stddev
    )
    grad = saliency_map(x_noise, model, class_index=class_index, func=func)
    return tf.reduce_mean(grad, axis=0, keepdims=True)


# ------------------------------------------------------------------------------


def integrated_grad(
    x, model, baseline, num_steps=25, class_index=None, func=tf.math.reduce_mean
):
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
    avg_grad = integral_approximation(grad)
    avg_grad = np.expand_dims(avg_grad, axis=0)
    return avg_grad


# ------------------------------------------------------------------------------


def expected_integrated_grad(
    x, model, baselines, num_steps=25, class_index=None, func=tf.math.reduce_mean
):
    """average integrated gradients across different backgrounds"""

    grads = []
    for baseline in baselines:
        grads.append(
            integrated_grad(
                x,
                model,
                baseline,
                num_steps=num_steps,
                class_index=class_index,
                func=tf.math.reduce_mean,
            )
        )
    return np.mean(np.array(grads), axis=0)


# ------------------------------------------------------------------------------


def mutagenesis(x, model, class_index=None):
    """in silico mutagenesis analysis for a given sequence"""

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

    # reshape mutagenesis predictiosn
    mut_score = reconstruct_map(predictions)

    return mut_score - wt_score


# ------------------------------------------------------------------------------


def grad_times_input(x, scores):
    new_scores = []
    for i, score in enumerate(scores):
        new_scores.append(np.sum(x[i] * score, axis=1))
    return np.array(new_scores)


def l2_norm(scores):
    return np.sum(np.sqrt(scores**2), axis=2)


# ------------------------------------------------------------------------------
# Useful functions
# ------------------------------------------------------------------------------


def function_batch(X, fun, batch_size=128, **kwargs):
    """run a function in batches"""

    dataset = tf.data.Dataset.from_tensor_slices(X)
    outputs = []
    for x in dataset.batch(batch_size):
        outputs.append(fun(x, **kwargs))
    return np.concatenate(outputs, axis=0)


def random_shuffle(x, num_samples=1):
    """randomly shuffle sequences
    assumes x shape is (N,L,A)"""

    x_shuffle = []
    for i in range(num_samples):
        shuffle = np.random.permutation(x.shape[1])
        x_shuffle.append(x[0, shuffle, :])
    return np.array(x_shuffle)


# ------------------------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------------------------


def grad_times_input_to_df(x, grad, alphabet="ACGT"):
    """generate pandas dataframe for saliency plot
    based on grad x inputs"""

    x_index = np.argmax(np.squeeze(x), axis=1)
    grad = np.squeeze(grad)
    L, A = grad.shape

    seq = ""
    saliency = np.zeros((L))
    for i in range(L):
        seq += alphabet[x_index[i]]
        saliency[i] = grad[i, x_index[i]]

    # create saliency matrix
    saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)
    return saliency_df
    
def grad_times_input_to_df_mutagenesis(x, grad, alphabet="ACGT"):
    import pandas as pd
    """Generate pandas dataframe for mutagenesis plot based on grad x inputs"""
    x = np.squeeze(x)  # Ensure x is correctly squeezed
    grad = np.squeeze(grad)
    L, A = x.shape

    # Get original nucleotides' indices, ensure it's 1D
    x_index = np.argmax(x, axis=1)
    
    # Convert index array to nucleotide letters
    original_nucleotides = np.array([alphabet[idx] for idx in x_index])

    # Data preparation for DataFrame
    data = {
        'Position': np.repeat(np.arange(L), A),
        'Nucleotide': np.tile(list(alphabet), L),
        'Effect': grad.reshape(-1),  # Flatten grad assuming it matches the reshaped size
        'Original': np.repeat(original_nucleotides, A)
    }
    df = pd.DataFrame(data)
    return df


def plot_attribution_map(saliency_df, ax=None, figsize=(20, 1)):
    """plot an attribution map using logomaker"""
    logomaker.Logo(saliency_df, figsize=figsize, ax=ax)
    if ax is None:
        ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    #ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")
    plt.xticks([])
    #plt.yticks([])

def plot_mutagenesis_map(mutagenesis_df, ax=None):
    """Plot an attribution map for mutagenesis using different colored dots, with adjusted x-axis limits."""
    colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
    if ax is None:
        ax = plt.gca()
    
    # Add horizontal line at y=0
    ax.axhline(0, color='gray', linewidth=1, linestyle='--')

    # Scatter plot for each nucleotide type
    for nuc, color in colors.items():
        # Filter out dots where the variant is the same as the original nucleotide
        subset = mutagenesis_df[(mutagenesis_df['Nucleotide'] == nuc) & (mutagenesis_df['Nucleotide'] != mutagenesis_df['Original'])]
        ax.scatter(subset['Position'], subset['Effect'], color=color, label=nuc, s=10)  # s is the size of the dot

    # Set the limits of the x-axis to match exactly the first and last position
    if not mutagenesis_df.empty:
        ax.set_xlim(mutagenesis_df['Position'].min() - 0.5, mutagenesis_df['Position'].max() + 0.5)

    ax.legend(title="Nucleotide", loc='upper right')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("none")
    plt.xticks([])  # Optionally, hide x-axis ticks for a cleaner look
