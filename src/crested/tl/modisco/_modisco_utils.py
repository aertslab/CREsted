from __future__ import annotations

import modiscolite as modisco
import numpy as np
import pandas as pd
from loguru import logger
from memelite import tomtom


def _trim_pattern_by_ic_old(
    pattern: dict,
    pos_pattern: bool,
    min_v: float,
    background: list[float] | None = None,
    pseudocount: float = 1e-6,
) -> dict:
    """
    Trims the pattern based on information content (IC).

    Parameters
    ----------
    pattern
        Dictionary containing the pattern data.
    pos_pattern
        Indicates if the pattern is a positive pattern.
    min_v
        Minimum value for trimming.
    background
        Background probabilities for each nucleotide.
    pseudocount
        Pseudocount for IC calculation.

    Returns
    -------
        Trimmed pattern.
    """
    if background is None:
        background = [0.28, 0.22, 0.22, 0.28]
    contrib_scores = np.array(pattern["contrib_scores"])
    if not pos_pattern:
        contrib_scores = -contrib_scores
    contrib_scores[contrib_scores < 0] = 1e-9  # avoid division by zero

    ic = modisco.util.compute_per_position_ic(
        ppm=np.array(contrib_scores), background=background, pseudocount=pseudocount
    )
    np.nan_to_num(ic, copy=False, nan=0.0)
    v = (abs(np.array(contrib_scores)) * ic[:, None]).max(1)
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)

    try:
        start_idx = min(np.where(np.diff((v > min_v) * 1))[0])
        end_idx = max(np.where(np.diff((v > min_v) * 1))[0]) + 1
    except ValueError:
        logger.error("No valid pattern found. Aborting...")

    return _trim(pattern, start_idx, end_idx)


def _trim_pattern_by_ic(
    pattern: dict,
    pos_pattern: bool,
    min_v: float,
    background: list[float] = None,
    pseudocount: float = 1e-6,
) -> dict:
    """
    Trims the pattern based on information content (IC).

    Parameters
    ----------
    pattern
        Dictionary containing the pattern data.
    pos_pattern
        Indicates if the pattern is a positive pattern.
    min_v
        Minimum value for trimming.
    background
        Background probabilities for each nucleotide.
    pseudocount
        Pseudocount for IC calculation.

    Returns
    -------
        Trimmed pattern.
    """
    if background is None:
        background = [0.28, 0.22, 0.22, 0.28]
    ppm = _pattern_to_ppm(pattern)

    _, ic, _ = compute_ic(ppm)
    np.nan_to_num(ic, copy=False, nan=0.0)
    v = (abs(ppm) * ic[:, None]).max(1)
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)

    try:
        if min_v > 0:
            start_idx = min(np.where(v > min_v)[0])
            end_idx = max(np.where(v > min_v)[0]) + 1
        else:
            start_idx = 0
            end_idx = len(ppm)

        if end_idx == start_idx:
            end_idx = start_idx + 1

        if end_idx == len(v):
            end_idx = len(v) - 1
    except ValueError:
        logger.error("No valid pattern found. Aborting...")

    return _trim(pattern, start_idx, end_idx)


def _trim(pattern: dict, start_idx: int, end_idx: int) -> dict:
    """
    Trims the pattern to the specified start and end indices.

    Parameters
    ----------
    pattern
        Dictionary containing the pattern data.
    start_idx
        Start index for trimming.
    end_idx (int)
        End index for trimming.

    Returns
    -------
        Trimmed pattern.
    """
    # TODO: Reading the pattern from disk should really be done in a seperate function!

    seqlet_dict = {}
    # read seqlet information
    for k in pattern["seqlets"].keys():
        seqlet_dict[k] = pattern["seqlets"][k][:]
    # do actual trimming
    seqlets_sequences = pattern["seqlets"]["sequence"]
    trimmed_sequences = [seq[start_idx:end_idx] for seq in seqlets_sequences]
    seqlet_dict["sequence"] = trimmed_sequences
    return {
        "sequence": np.array(pattern["sequence"])[start_idx:end_idx],
        "contrib_scores": np.array(pattern["contrib_scores"])[start_idx:end_idx],
        "hypothetical_contribs": np.array(pattern["hypothetical_contribs"])[
            start_idx:end_idx
        ],
        "seqlets": seqlet_dict,
    }


def _get_ic(
    contrib_scores: np.ndarray,
    pos_pattern: bool,
    background: list[float] | None = None,
) -> np.ndarray:
    """
    Compute the information content (IC) for the given contribution scores.

    Parameters
    ----------
    contrib_scores
        Array of contribution scores.
    pos_pattern
        Indicates if the pattern is a positive pattern.
    background
        background probabilities for each nucleotide.

    Returns
    -------
    Information content for the contribution scores.
    """
    if background is None:
        background = [0.27, 0.23, 0.23, 0.27]
    background = np.array(background)
    if not pos_pattern:
        contrib_scores = -contrib_scores
    contrib_scores[contrib_scores < 0] = 1e-9
    ppm = contrib_scores / np.sum(contrib_scores, axis=1)[:, None]

    ic = (np.log((ppm + 0.001) / (1.004)) / np.log(2)) * ppm - (
        np.log(background) * background / np.log(2)
    )
    return ppm * (np.sum(ic, axis=1)[:, None])


def _one_hot_to_count_matrix(sequences):
    """
    Convert a set of one-hot encoded sequences to a count matrix.

    Parameters
    ----------
    sequences
        A numpy array of shape (n_sequences, sequence_length, 4), representing the one-hot encoded sequences.

    Returns
    -------
    count_matrix
        A count matrix of shape (sequence_length, 4) where each entry represents the count of A, C, G, or T at each position.
    """
    # Sum the one-hot encoded sequences along the first axis (the sequence axis)
    count_matrix = np.sum(sequences, axis=0)

    return count_matrix


def _count_matrix_to_ppm(count_matrix, pseudocount=1.0):
    """
    Convert a count matrix to a position weight matrix (PWM) by adding pseudocounts and normalizing by the total counts per position.

    Parameters
    ----------
    count_matrix
        A count matrix of shape (sequence_length, 4).
    pseudocount
        A pseudocount added to each nucleotide count to avoid zeros.

    Returns
    -------
    pwm
        The position weight matrix of shape (sequence_length, 4).
    """
    # Add pseudocount to avoid zero probabilities
    count_matrix += pseudocount

    # Calculate the total count at each position (sum across nucleotide axis)
    total_counts = np.sum(count_matrix, axis=1, keepdims=True)

    # Normalize to get PWM (divide each count by the total counts at that position)
    ppm = count_matrix / total_counts

    return ppm


def _ppm_to_pwm(ppm, background_frequencies=None):
    """
    Convert a Position Probability Matrix (PPM) to a Position Weight Matrix (PWM) using log-odds.

    Parameters
    ----------
    ppm
        A PPM of shape (sequence_length, 4) where each value is a probability.
    background_frequencies
        Background frequencies for A, C, G, T. Default is [0.27, 0.23, 0.23, 0.27].

    Returns
    -------
    pwm
        The Position Weight Matrix of shape (sequence_length, 4).
    """
    if background_frequencies is None:
        # Uniform background frequencies for A, C, G, T
        background_frequencies = np.array([0.28, 0.22, 0.22, 0.28])
    else:
        background_frequencies = np.array(background_frequencies)

    # Ensure no division by zero or log(0) by replacing 0s in the PPM with a small value
    ppm = np.clip(ppm, 1e-3, None)

    # Apply log-odds transformation to convert PPM to PWM
    pwm = np.log2(ppm / background_frequencies)

    return pwm


def _pattern_to_ppm(pattern):
    seqs = np.array(pattern["seqlets"]["sequence"])
    count_matrix = _one_hot_to_count_matrix(seqs)
    ppm = _count_matrix_to_ppm(count_matrix)
    return ppm


def compute_ic(ppm, background_freqs: list | None = None):
    """
    Compute the information content (IC) of a Position Probability Matrix (PPM).

    Parameters
    ----------
    ppm
        2D numpy array where rows correspond to positions in the motif, and columns correspond to symbols (A, T, C, G for example).
    background_freqs
        1D numpy array with the background frequencies of each symbol.

    Returns
    -------
    total_ic
        Total information content of the PPM.
    ic_per_position
        Information content per position in the motif.
    ic_per_element
        2D array of information content per element in the PPM.
    """
    # Ensure ppm is a numpy array
    if background_freqs is None:
        background_freqs = [0.28, 0.22, 0.22, 0.28]
    ppm = np.array(ppm)
    background_freqs = np.array(background_freqs)

    # Initialize the IC array for each element
    ic_per_element = np.zeros_like(ppm)

    # Calculate the IC per element using the formula -p_ij * log2(p_ij / p_j)
    for i in range(ppm.shape[0]):  # for each position in the motif
        for j in range(ppm.shape[1]):  # for each symbol (A, T, C, G)
            if ppm[i, j] > 0:  # Avoid log(0)
                ic_per_element[i, j] = ppm[i, j] * np.log2(
                    ppm[i, j] / background_freqs[j]
                )

    # IC per position is the sum of IC values across symbols at each position
    ic_per_position = np.sum(ic_per_element, axis=1)

    # Total IC is the sum of IC values across all positions
    total_ic = np.sum(ic_per_element)

    return total_ic, ic_per_position, ic_per_element


def l1(X: np.ndarray) -> np.ndarray:
    """
    Normalize the input array using the L1 norm.

    Parameters
    ----------
    X
        Input array.

    Returns
    -------
    L1 normalized array.
    """
    abs_sum = np.sum(np.abs(X))
    return X if abs_sum == 0 else (X / abs_sum)


def get_2d_data_from_patterns(
    pattern: dict, transformer: str = "l1", include_hypothetical: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get 2D data from patterns using specified transformer.

    Parameters
    ----------
    pattern
        Dictionary containing pattern data.
    transformer
        Transformer function to use ('l1' or 'magnitude').
    include_hypothetical
        Whether to include hypothetical contributions.

    Returns
    -------
    Forward and reverse 2D data arrays.
    """
    func = l1 if transformer == "l1" else None  # magnitude not defined?
    tracks = (
        ["hypothetical_contribs", "contrib_scores"]
        if include_hypothetical
        else ["contrib_scores"]
    )

    all_fwd_data, all_rev_data = [], []
    snippets = [pattern[track] for track in tracks]

    fwd_data = np.concatenate([func(snippet) for snippet in snippets], axis=1)
    rev_data = np.concatenate(
        [func(snippet[::-1, ::-1]) for snippet in snippets], axis=1
    )

    all_fwd_data.append(fwd_data)
    all_rev_data.append(rev_data)

    return np.array(all_fwd_data), np.array(all_rev_data)


def pad_pattern(pattern: dict, pad_len: int = 2) -> dict:
    """
    Pad the pattern with zeros.

    Parameters
    ----------
    pattern
        dictionary containing the pattern data.
    pad_len
        Length of padding.

    Returns
    -------
    Padded pattern.
    """
    p0 = pattern.copy()
    p0["contrib_scores"] = np.concatenate(
        (np.zeros((pad_len, 4)), p0["contrib_scores"], np.zeros((pad_len, 4)))
    )
    p0["hypothetical_contribs"] = np.concatenate(
        (np.zeros((pad_len, 4)), p0["hypothetical_contribs"], np.zeros((pad_len, 4)))
    )
    return p0


def match_score_patterns(
        a: list[dict] | dict,
        b: list[dict] | dict,
        use_ppm: bool = False,
        background_freqs: list | None = None,
    ) -> float | np.ndarray:
    """
    Compute the match score between two sets of patterns using TOMTOM through memesuite-lite.

    Parameters
    ----------
    a
        First set of patterns.
    b
        Second set of patterns.
    use_ppm
        Use PPM instead of PWM for TOMTOM comparison.
    background_freqs
        1D numpy array with the background frequencies of each symbol.

    Returns
    -------
    Match TOMTOM score (-log10(pval)) between the patterns. Return a float if it is a one vs one comparison, a 2D numpy array when comparing lists of motifs.
    """
    if background_freqs is None:
        background_freqs = [0.28, 0.22, 0.22, 0.28]
    background_freqs = np.array(background_freqs)

    if not isinstance(a, list):
        a = [a]
    if not isinstance(b, list):
        b = [b]

    if not use_ppm:
        a = [compute_ic(pat["ppm"], background_freqs=background_freqs)[2].T for pat in a]
        b = [compute_ic(pat["ppm"], background_freqs=background_freqs)[2].T for pat in b]
    else:
        a = [pat["ppm"].T for pat in a]
        b = [pat["ppm"].T for pat in b]

    try:
        p, _, _, _, _ = tomtom(Qs=a, Ts=b)

    except Exception as e:  # noqa: BLE001
        print(
            "Warning: TOMTOM error while comparing patterns. Returning no match."
        )
        print(f"Error details: {e}")
        p = np.ones((len(a), len(b)))

    p[p<=0]=1e-15 # Sometimes negative value returned

    log_score = -np.log10(p)

    if log_score.shape == (1, 1):# Return a float if it is only a one vs one comparison
        return log_score[0,0]

    return log_score


def _match_score_patterns_old(a: dict, b: dict) -> float:
    """
    Compute the match score between two patterns.

    Parameters
    ----------
    a
        First pattern.
    b
        Second pattern.

    Returns
    -------
    Match score between the patterns.
    """
    a = pad_pattern(a)
    fwd_data_A, rev_data_A = get_2d_data_from_patterns(a)
    fwd_data_B, rev_data_B = get_2d_data_from_patterns(b)
    X = fwd_data_B if fwd_data_B.shape[1] <= fwd_data_A.shape[1] else fwd_data_A
    Y = fwd_data_A if fwd_data_B.shape[1] <= fwd_data_A.shape[1] else fwd_data_B
    sim_fwd_pattern = np.array(modisco.affinitymat.jaccard(X, Y).squeeze())
    X = fwd_data_B if fwd_data_B.shape[1] <= fwd_data_A.shape[1] else rev_data_A
    Y = rev_data_A if fwd_data_B.shape[1] <= fwd_data_A.shape[1] else fwd_data_B
    sim_rev_pattern = np.array(modisco.affinitymat.jaccard(X, Y).squeeze())

    return max(sim_fwd_pattern[0], sim_rev_pattern[0])


def read_html_to_dataframe(source: str):
    """
    Read an HTML table from the Modisco report function into a DataFrame.

    Parameters
    ----------
    source
        The URL or file path to the HTML content.

    Returns
    -------
    DataFrame containing the HTML table or an error message if no table is found.
    """
    try:
        # Attempt to read the HTML content
        dfs = pd.read_html(source)

        # Check if any tables were found
        if not dfs:
            return "No tables found in the HTML content."

        # Return the first DataFrame
        return dfs[0]
    except ValueError as e:
        # Handle the case where no tables are found
        return f"Error: {str(e)}"


def write_to_meme(ppms: dict, output_file: str):
    """
    Write PPMs to a MEME-format file.

    Parameters
    ----------
    ppms
        Dictionary of PPMs where keys are pattern IDs and values are numpy arrays.
    output_file
        Path to the output MEME file.
    """
    with open(output_file, "w") as f:
        # Write the MEME header
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies:\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

        # Write each pattern in MEME format
        for pattern_id, ppm in ppms.items():
            motif_length, _ = ppm.shape
            f.write(f"MOTIF {pattern_id}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {motif_length}\n")
            for row in ppm:
                f.write(f"{' '.join(f'{x:.6f}' for x in row)}\n")
            f.write("\n")

    print(f"PPMs saved to {output_file} in MEME format.")
