"""Dataloader for batching, shuffling, and one-hot encoding of AnnDataset."""

from __future__ import annotations

import os
from collections import defaultdict

if os.environ["KERAS_BACKEND"] == "torch":
    import torch
    from torch.utils.data import DataLoader, Sampler
else:
    import tensorflow as tf

from ._dataset import AnnDataset


from torch.utils.data import Sampler
import numpy as np
 
class WeightedRegionSampler(Sampler):
    def __init__(self, dataset, epoch_size=100_000):
        super().__init__(data_source=dataset)
        self.dataset = dataset
        self.epoch_size = epoch_size
        p = dataset.augmented_probs
        s = p.sum()
        if s <= 0:
            raise ValueError("All sample_prob are zero, cannot sample.")
        self.probs = p / s

    def __iter__(self):
        n = len(self.dataset.index_manager.augmented_indices)
        for _ in range(self.epoch_size):
            yield np.random.choice(n, p=self.probs)

    def __len__(self):
        return self.epoch_size

class NonShuffleRegionSampler(Sampler):
    """
    Enumerate each region with sample_prob>0 exactly once, in a deterministic order.
    """

    def __init__(self, dataset):
        super().__init__(data_source=dataset)
        self.dataset = dataset

        # We get the augmented_probs from dataset.augmented_probs
        # We'll filter out any zero-prob entries
        p = self.dataset.augmented_probs
        self.nonzero_indices = np.flatnonzero(p > 0.0)  # e.g. [0,1,5,...]
        if len(self.nonzero_indices) == 0:
            raise ValueError("No nonzero probabilities for val/test stage.")

    def __iter__(self):
        # Return each index once, in ascending order
        # or sort by some custom logic
        return iter(self.nonzero_indices)

    def __len__(self):
        # The DataLoader sees how many samples in an epoch
        return len(self.nonzero_indices)

class MetaSampler(Sampler):
    """
    A Sampler that yields indices in proportion to meta_dataset.global_probs.
    """

    def __init__(self, meta_dataset: MetaAnnDataset, epoch_size: int = 100_000):
        """
        Parameters
        ----------
        meta_dataset : MetaAnnDataset
            The combined dataset with global_indices and global_probs.
        epoch_size : int
            How many samples we consider in one epoch of training.
        """
        super().__init__(data_source=meta_dataset)
        self.meta_dataset = meta_dataset
        self.epoch_size = epoch_size

        # Check that sum of global_probs ~ 1.0
        s = self.meta_dataset.global_probs.sum()
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(
                "global_probs do not sum to 1 after final normalization. sum = {}".format(s)
            )

    def __iter__(self):
        """
        For each epoch, yield 'epoch_size' random draws from
        [0..len(meta_dataset)-1], weighted by global_probs.
        """
        n = len(self.meta_dataset)
        p = self.meta_dataset.global_probs
        for _ in qrange(self.epoch_size):
            yield np.random.choice(n, p=p)

    def __len__(self):
        """
        The DataLoader uses len(sampler) to figure out how many samples per epoch.
        """
        return self.epoch_size

class NonShuffleMetaSampler(Sampler):
    """
    A Sampler for MetaAnnDataset that enumerates all indices
    with nonzero global_probs exactly once, in ascending order.

    Typically used for val/test phases, ensuring deterministic
    coverage of all relevant entries.
    """

    def __init__(self, meta_dataset, sort=True):
        """
        Parameters
        ----------
        meta_dataset : MetaAnnDataset
            The combined dataset with .global_indices and .global_probs.
        sort : bool
            If True, sort the nonzero indices ascending. If False, keep them in
            the existing order. You can also implement your own custom ordering.
        """
        super().__init__(data_source=meta_dataset)
        self.meta_dataset = meta_dataset

        # We'll gather the set of global indices with probability > 0
        p = self.meta_dataset.global_probs
        self.nonzero_global_indices = np.flatnonzero(p > 0)
        if sort:
            self.nonzero_global_indices.sort()

    def __iter__(self):
        # yields each global index exactly once
        return iter(self.nonzero_global_indices)

    def __len__(self):
        return len(self.nonzero_global_indices)

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
        stage: str = "train",
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

        if isinstance(dataset, MetaAnnDataset):
            # If this is the training stage => random draws
            if stage == "train":
                self.sampler = MetaSampler(dataset, epoch_size=self.epoch_size)
            else:
                # e.g. val or test => enumerates all nonzero-prob entries once
                self.sampler = NonShuffleMetaSampler(dataset, sort=True)
        else:
            # Single AnnDataset => check stage
            if dataset.augmented_probs is not None:
                if stage == "train":
                    # Weighted random draws
                    self.sampler = WeightedRegionSampler(dataset, epoch_size=self.epoch_size)
                else:
                    # val/test => enumerates nonzero-prob entries once
                    self.sampler = NonShuffleRegionSampler(dataset)
            else:
                # uniform approach
                if shuffle and hasattr(self.dataset, "shuffle"):
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
