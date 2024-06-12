import logomaker
import numpy as np


def grad_times_input_to_df(x, grad, alphabet="ACGT"):
    """Generate pandas dataframe for saliency plot based on grad x inputs"""
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
        "Position": np.repeat(np.arange(L), A),
        "Nucleotide": np.tile(list(alphabet), L),
        "Effect": grad.reshape(
            -1
        ),  # Flatten grad assuming it matches the reshaped size
        "Original": np.repeat(original_nucleotides, A),
    }
    df = pd.DataFrame(data)
    return df
