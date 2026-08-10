"""AnnDataWrapper class to load sequences and AnnData values from your genome and AnnData of choice."""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from loguru import logger
from scipy.sparse import spmatrix

from crested._genome import Genome

from .utils import BaseGenomicDataWrapper


class AnnDataWrapper(BaseGenomicDataWrapper):
    """
    Wrapper around your AnnData and genome, providing you with one-hot encoded sequences and associated scalar values to train a model with.

    The :obj:`~crested.tl.Crested` class expects this or :obj:`~crested.tl.data.AnnDataModule`.

    Parameters
    ----------
    data
        Your AnnData object.
    genome
        The genome to extract sequences from, as a crested.Genome object. If None, will look up the registered Genome.
    batch_size
        Batch size to use during training and evaluation.
    random_reverse_complement
        If True, the sequences will be randomly reverse complemented during training.
        Incompatible with always_reverse_complement.
    always_reverse_complement
        If True, the dataset will be expanded to include both the forward and reverse-complemented versions of every entry in the training set.
        Incompatible with random_reverse_complement.
    max_stochastic_shift
        Maximum stochastic shift (n base pairs in either direction) to apply randomly to each sequence during training.
        Default is 0 (disabled).
    drop_remainder
        If True, drop the last batch if it is not the full batch_size. Default is False.
    train_splits
        The values in your split labeling that correspond to the training set as string or list of strings, i.e 'train' or ['fold0', 'fold1', 'fold2']
        If None, uses the values that aren't `val_splits` or `test_splits`.
    val_splits
        The values in your split labeling that correspond to the validation set as string or list of strings, i.e 'val' or ['fold3', 'fold4']
    test_splits
        The values in your split labeling that correspond to the test set as string or list of strings, i.e 'test' or ['fold5', 'fold6']
    split_column
        The column in adata.var that contains the values to split on (as provided to [train/val/test]_splits)
    kwargs
        Arguments passed to :obj:`~crested.tl.data.utils.BaseGenomicDataWrapper`.
    """

    def __init__(
        self,
        data: AnnData,
        genome: Genome | None = None,
        batch_size: int = 256,
        random_reverse_complement: bool = False,
        always_reverse_complement: bool = True,
        max_stochastic_shift: int = 0,
        in_memory: bool = True,
        drop_remainder: bool = False,
        train_splits: str | list | None = None,
        val_splits: str | list = 'val',
        test_splits: str | list = 'test',
        split_column: str = 'split',
        **kwargs
    ):
        """Initialize the AnnDataWrapper with an AnnData and a genome."""
        # Set some basic values (esp those required for _get_indices and _get_splits)
        self.data = data
        self.split_column = split_column
        self.compressed = isinstance(self.data.X, spmatrix)

        # Initialize base genomicdatawrapper functionality (creating indices and interfacing with the genome)
        super().__init__(
            genome=genome,
            batch_size=batch_size,
            random_reverse_complement=random_reverse_complement,
            always_reverse_complement=always_reverse_complement,
            max_stochastic_shift=max_stochastic_shift,
            in_memory=in_memory,
            drop_remainder=drop_remainder,
            train_splits=train_splits,
            val_splits=val_splits,
            test_splits=test_splits,
            **kwargs
        )

        # Set some last variables dependent on having indices or extracting sequences
        self.index_map = {index: i for i, index in enumerate(self.indices)}

    def _get_indices(self):
        """Return a full list of all included sample indices, aka the anndata's var_names."""
        return list(self.data.var_names)

    def _get_splits(self):
        """Return a list of split values, for each index from _get_indices()."""
        return list(self.data.var[self.split_column])

    def _get_target(self, original_index: str, **kwargs) -> np.ndarray:
        """Get target for a given index. Returned value should not have a batch dimension yet.

        If not using certain arguments in your implementation (like only using one of original_index/expanded_index), please keep **kwargs to absorb the un-used other arguments.

        Parameters
        ----------
        original_index
            The original index of the sequence, as present in the anndata's var_names.
        kwargs
            Catcher for unused arguments from `get_indexed_item`, specifically `expanded_index`, `revcomp`, and `shift`.
        """
        y_index = self.index_map[original_index]
        return (
            self.data.X[:, y_index].toarray().flatten()
            if self.compressed
            else self.data.X[:, y_index].astype('float32')
        )

    def get_config(self) -> dict:
        """Return a dict of properties, to be logged during training.

        Primarily used in Crested.fit().
        """
        config = super().get_config()
        config.update({
            "split_column": self.split_column,
            "compressed": self.compressed
        })
        return config

class MultiAnnDataWrapper(BaseGenomicDataWrapper):
    """
    Wrapper around multiple AnnDatas and a genome, providing you with one-hot encoded sequences and multiple sets of associated scalar values to train a model with.

    This assumes the AnnDatas contain the same inputs (regions or genes), and that they simply have different class values (same .var, different .obs).
    The first AnnData will be used to create the index, and all other AnnDatas must contain those indices as well.
    Required input for the `tl.Crested` class.

    Parameters
    ----------
    data
        List of your AnnData objects.
    genome
        The genome to extract sequences from, as a crested.Genome object. If None, will look up the registered Genome.
    batch_size
        Batch size to use during training and evaluation.
    random_reverse_complement
        If True, the sequences will be randomly reverse complemented during training.
        Incompatible with always_reverse_complement.
    always_reverse_complement
        If True, the dataset will be expanded to include both the forward and reverse-complemented versions of every entry in the training set.
        Incompatible with random_reverse_complement.
    max_stochastic_shift
        Maximum stochastic shift (n base pairs in either direction) to apply randomly to each sequence during training.
        Default is 0 (disabled).
    drop_remainder
        If True, drop the last batch if it is not the full batch_size. Default is False.
    train_splits
        The values in your split labeling that correspond to the training set as string or list of strings, i.e 'train' or ['fold0', 'fold1', 'fold2']
        If None, uses the values that aren't `val_splits` or `test_splits`.
    val_splits
        The values in your split labeling that correspond to the validation set as string or list of strings, i.e 'val' or ['fold3', 'fold4']
    test_splits
        The values in your split labeling that correspond to the test set as string or list of strings, i.e 'test' or ['fold5', 'fold6']
    split_column
        The column in adata.var that contains the values to split on (as provided to [train/val/test]_splits)
    """

    def __init__(
        self,
        data: list[AnnData],
        genome: Genome | None = None,
        batch_size: int = 256,
        random_reverse_complement: bool = False,
        always_reverse_complement: bool = True,
        max_stochastic_shift: int = 0,
        in_memory: bool = True,
        drop_remainder: bool = False,
        train_splits: str | list | None = None,
        val_splits: str | list = 'val',
        test_splits: str | list = 'test',
        split_column: str = 'split',
        **kwargs
    ):
        """Initialize the MultiAnnDataWrapper with AnnDatas and a genome."""
        # Set some basic values (esp those required for _get_indices and _get_splits)
        self.data = data
        self.split_column = split_column
        self.compressed = [isinstance(self.data[i].X, spmatrix) for i in range(self.n_datasets)]
        for i in range(1, self.n_datasets):
            if self.data[i].n_vars != self.data[0].n_vars:
                logger.warning(f"AnnData {i}'s number of regions ({self.data[i].n_vars}) doesn't match the first AnnData's number of regions ({self.data[0].n_vars}). MultiAnnDataWrapper requires all AnnDatas to contain at least the regions of the first AnnData, so make sure this AnnData contains a superset of those indices.")

        # Initialize base genomicdatawrapper functionality (creating indices and interfacing with the genome)
        super().__init__(
            genome=genome,
            batch_size=batch_size,
            random_reverse_complement=random_reverse_complement,
            always_reverse_complement=always_reverse_complement,
            max_stochastic_shift=max_stochastic_shift,
            in_memory=in_memory,
            drop_remainder=drop_remainder,
            train_splits=train_splits,
            val_splits=val_splits,
            test_splits=test_splits,
            **kwargs
        )

        # Build index map for original anndata (used to index into .X to get ground truth)
        self.index_maps = [{index: i for i, index in enumerate(self.indices)}]

        # Add index map for other anndatas and check whether they all contain the original indices
        for adata_i in range(1, self.n_datasets):
            adata_index_list = self._get_indices(index=adata_i)
            self.index_maps.append({index: i for i, index in enumerate(adata_index_list)})
            assert all(main_adata_index in adata_index_list for main_adata_index in self.indices), f"AnnData {adata_i} does not have all indices contained in the first AnnData. First AnnData shape: {self.data[0].shape}, offending AnnData shape: {self.data[adata_i].shape}"

    def _get_indices(self, index=0):
        """Return a full list of all included sample indices, aka the anndata's var_names."""
        return list(self.data[index].var_names)

    def _get_splits(self):
        """Return a list of split values, for each index from _get_indices()."""
        return list(self.data[0].var[self.split_column])

    def _get_target(self, original_index: str, **kwargs) -> np.ndarray:
        """Get target for a given index. Returned value should not have a batch dimension yet.

        If not using certain arguments in your implementation (like only using one of original_index/expanded_index), please keep **kwargs to absorb the un-used other arguments.

        Parameters
        ----------
        original_index
            The original index of the sequence, as present in the anndata's var_names.
        kwargs
            Catcher for unused arguments from `get_indexed_item`, specifically `expanded_index`, `revcomp`, and `shift`.
        """
        return tuple(
            (
                self.data[i].X[:, self.index_maps[i][original_index]].toarray().flatten()
                if self.compressed[i]
                else self.data[i].X[:, self.index_maps[i][original_index]].astype('float32')
            ) for i in range(self.n_datasets)
        )

    def get_config(self) -> dict:
        """Return a dict of properties, to be logged during training.

        Primarily used in Crested.fit().
        """
        config = super().get_config()
        config.update({
            "split_column": self.split_column,
            "compressed": self.compressed
        })
        return config

    @property
    def n_datasets(self):
        """Return the number of datasets in the MultiAnnDataWrapper."""
        return len(self.data)
