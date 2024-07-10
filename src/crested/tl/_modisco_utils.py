from __future__ import annotations

import modiscolite as modisco
import numpy as np
import pandas as pd


def l1(X: np.ndarray) -> np.ndarray:
    """
    Normalizes the input array using the L1 norm.

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
    Gets 2D data from patterns using specified transformer.

    Parameters
    ----------
    - pattern (dict): Dictionary containing pattern data.
    - transformer (str): Transformer function to use ('l1' or 'magnitude').
    - include_hypothetical (bool): Whether to include hypothetical contributions.

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
    Pads the pattern with zeros.

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


def match_score_patterns(a: dict, b: dict) -> float:
    """
    Computes the match score between two patterns.

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
    Reads an HTML table from the Modisco report function into a DataFrame.

    Parameters
    ----------
    - source: str - The URL or file path to the HTML content.

    Returns
    -------
    - DataFrame containing the HTML table or an error message if no table is found.
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
