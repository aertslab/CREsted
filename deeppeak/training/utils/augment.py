"""Functions for online data augmentations during model training."""

import tensorflow as tf


def complement_base(X: tf.Tensor) -> tf.Tensor:
    """Complement a DNA sequence.

    Args:
        X: A tensor of shape (seq_len, 4) representing a batch of DNA
        sequences.

    Returns:
        A tensor of shape (seq_len, 4) representing
        the complemented DNA sequences.
    """
    do_augment = tf.random.uniform([]) < 0.5
    X = tf.cond(do_augment, lambda: X[::-1, ::-1], lambda: X)
    return X
