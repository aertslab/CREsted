"""Dataloader for batching, shuffling, and one-hot encoding of AnnDataset."""

from __future__ import annotations

import os
from collections import defaultdict

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
        dataset,  # can be AnnDataset or MetaAnnDataset
        batch_size: int,
        shuffle: bool = False,
        drop_remainder: bool = True,
        epoch_size: int = 100_000,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.epoch_size = epoch_size

        if os.environ.get("KERAS_BACKEND", "") == "torch":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

        self.sampler = None

        # Decide if we should use MetaSampler
        if isinstance(dataset, MetaAnnDataset):
            # This merges many AnnDataset objects, so let's use MetaSampler
            self.sampler = MetaSampler(dataset, epoch_size=self.epoch_size)
        else:
            # Single AnnDataset => possibly fallback to WeightedRegionSampler or uniform
            # We'll do uniform shuffle if asked. WeightedRegionSampler is not shown here,
            # but you could do:
            # if dataset.augmented_probs is not None: self.sampler = WeightedRegionSampler(...)
            if self.shuffle and hasattr(self.dataset, "shuffle"):
                self.dataset.shuffle = True

    def _collate_fn(self, batch):
        """
        Collate function to gather list of sample-dicts into a single batched dict of tensors.
        """
        x = defaultdict(list)
        for sample_dict in batch:
            for key, val in sample_dict.items():
                x[key].append(torch.tensor(val, dtype=torch.float32))

        # Stack and move to device
        for key in x.keys():
            x[key] = torch.stack(x[key], dim=0)
            if self.device is not None:
                x[key] = x[key].to(self.device)
        return x

    def _create_dataset(self):
        if os.environ.get("KERAS_BACKEND", "") == "torch":
            if self.sampler is not None:
                return DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    sampler=self.sampler,
                    drop_last=self.drop_remainder,
                    num_workers=0,
                    collate_fn=self._collate_fn,
                )
            else:
                return DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
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
        if self.sampler is not None:
            return (self.epoch_size + self.batch_size - 1) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __repr__(self):
        """Return the string representation of the DataLoader."""
        return (
            f"AnnDataLoader(dataset={self.dataset}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_remainder={self.drop_remainder})"
        )
