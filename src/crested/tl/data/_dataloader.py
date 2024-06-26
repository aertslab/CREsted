"""Dataloader for batching, shuffling, and one-hot encoding of AnnDataset."""

from __future__ import annotations

import tensorflow as tf

from ._dataset import AnnDataset


class AnnDataLoader:
    """
    Pytorch-like DataLoader class for AnnDataset with options for batching, shuffling, and one-hot encoding.

    Parameters
    ----------
    dataset
        The dataset instance provided.
    batch_size
        Number of samples per batch.
    shuffle
        Indicates whether shuffling is enabled.
    drop_remainder
        Indicates whether to drop the last incomplete batch.

    Examples
    --------
    >>> dataset = AnnDataset(...)  # Your dataset instance
    >>> batch_size = 32
    >>> dataloader = AnnDataLoader(
    ...     dataset, batch_size, shuffle=True, drop_remainder=True
    ... )
    >>> for x, y in dataloader.data:
    ...     # Your training loop here
    """

    def __init__(
        self,
        dataset: AnnDataset,
        batch_size: int,
        shuffle: bool = False,
        drop_remainder: bool = True,
    ):
        """Initialize the DataLoader with the provided dataset and options."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder

        if self.shuffle:
            self.dataset.shuffle = True

    def _create_dataset(self):
        ds = tf.data.Dataset.from_generator(
            self.dataset,
            output_signature=(
                tf.TensorSpec(shape=(self.dataset.seq_len, 4), dtype=tf.float16),
                tf.TensorSpec(shape=(self.dataset.num_outputs,), dtype=tf.float32),
            ),
        )
        ds = (
            ds.batch(self.batch_size, drop_remainder=self.drop_remainder)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

    @property
    def data(self):
        """Return the dataset as a tf.data.Dataset instance."""
        return self._create_dataset()

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __repr__(self):
        return (
            f"AnnDataLoader(dataset={self.dataset}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, one_hot_encode={self.one_hot_encode}, "
            f"drop_remainder={self.drop_remainder})"
        )
