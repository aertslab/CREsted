"""Dataloader for batching, shuffling, and one-hot encoding of AnnDataset."""

from __future__ import annotations

import os

if os.environ["KERAS_BACKEND"] == "torch":
    import torch
    from torch.utils.data import DataLoader
else:
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
        if os.environ["KERAS_BACKEND"] == "torch":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.shuffle:
            self.dataset.shuffle = True

    def _collate_fn(self, batch):
        """Collate function to move tensors to the specified device if backend is torch."""
        inputs, targets = zip(*batch)
        inputs = torch.stack([torch.tensor(input) for input in inputs]).to(self.device)
        targets = torch.stack([torch.tensor(target) for target in targets]).to(
            self.device
        )
        return inputs, targets

    def _create_dataset(self):
        if os.environ["KERAS_BACKEND"] == "torch":
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                drop_last=self.drop_remainder,
                num_workers=0,
                collate_fn=self._collate_fn,
            )
        elif os.environ["KERAS_BACKEND"] == "tensorflow":
            ds = tf.data.Dataset.from_generator(
                self.dataset,
                output_signature=(
                    tf.TensorSpec(shape=(self.dataset.seq_len, 4), dtype=tf.float32),
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
        """Return the number of batches in the DataLoader based on the dataset size and batch size."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __repr__(self):
        """Return the string representation of the DataLoader."""
        return (
            f"AnnDataLoader(dataset={self.dataset}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_remainder={self.drop_remainder})"
        )
