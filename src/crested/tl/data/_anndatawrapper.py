"""AnnDataWrapper class to load sequences and AnnData values from your genome and AnnData of choice."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import spmatrix

from crested._genome import Genome
from crested.utils import one_hot_encode_sequence

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
    gene_neighbors
        Optional path to a gene-neighbors TSV (as produced by the `gene_window/gene_neighbors.py`
        companion script, with `name`, `prev_gene_end`, `next_gene_start` columns) or an
        already-loaded DataFrame in that shape.
        If provided, a 5th channel is appended to the one-hot encoded sequence, marking positions
        within this gene's own territory (its body plus the intergenic buffer up to the
        neighboring genes) as 1.0, and anything else (e.g. a neighboring gene's own body, reached
        via stochastic shifting) as 0.0. Assumes each var's region is exactly the gene body itself.
        If None (default), no channel is added and behavior/shape are unchanged.
    gene_id_column
        The column in `adata.var` holding the gene identifier to match against `gene_neighbors`'
        `name` column. Only used if `gene_neighbors` is provided.
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
        gene_neighbors: str | os.PathLike | pd.DataFrame | None = None,
        gene_id_column: str = "gene_id",
        **kwargs
    ):
        """Initialize the AnnDataWrapper with an AnnData and a genome."""
        # Set some basic values (esp those required for _get_indices and _get_splits)
        self.data = data
        self.split_column = split_column
        self.compressed = isinstance(self.data.X, spmatrix)

        # Load and validate gene-neighbors table, if provided (before super().__init__ since it uses self.data)
        self.gene_id_column = gene_id_column
        self.gene_neighbors_path = gene_neighbors if isinstance(gene_neighbors, (str, os.PathLike)) else None
        self._gene_neighbors = (
            self._load_gene_neighbors(gene_neighbors, gene_id_column) if gene_neighbors is not None else None
        )

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

    def _load_gene_neighbors(self, gene_neighbors: str | os.PathLike | pd.DataFrame, gene_id_column: str) -> pd.DataFrame:
        """Load and validate the gene-neighbors annotation table, indexed by gene id."""
        df = gene_neighbors if isinstance(gene_neighbors, pd.DataFrame) else pd.read_csv(gene_neighbors, sep="\t")

        required_cols = {"name", "prev_gene_end", "next_gene_start"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"`gene_neighbors` is missing required column(s): {sorted(missing_cols)}.")
        if df["name"].duplicated().any():
            raise ValueError("`gene_neighbors`'s 'name' column must be unique.")
        if gene_id_column not in self.data.var.columns:
            raise ValueError(
                f"`gene_id_column` '{gene_id_column}' not found in adata.var. "
                f"Available columns: {list(self.data.var.columns)}"
            )

        df = df.set_index("name")[["prev_gene_end", "next_gene_start"]]

        missing_genes = sorted(set(self.data.var[gene_id_column]) - set(df.index))
        if missing_genes:
            raise ValueError(
                f"{len(missing_genes)} gene id(s) from adata.var['{gene_id_column}'] not found in "
                f"`gene_neighbors`, e.g. {missing_genes[:5]}."
            )
        return df

    def _get_gene_territory_mask(
        self,
        original_index: str,
        parsed_index: tuple[str, int, int, str],
        shift: int = 0,
        revcomp: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Boolean mask (seq_len,) aligned to the sequence returned by `_get_sequence`.

        True marks positions within this gene's own territory: its annotated body plus the
        intergenic buffer up to (but not including) the neighboring genes from `gene_neighbors`.
        False marks positions that, after stochastic shifting, fall onto a neighboring gene's own
        body. Mirrors the shift/strand/revcomp handling of
        `BaseGenomicDataWrapper._get_shuffle_mask`.
        """
        chrom, start, end, strand = parsed_index
        seq_len = end - start

        gene_id = self.data.var.loc[original_index, self.gene_id_column]
        prev_gene_end, next_gene_start = self._gene_neighbors.loc[gene_id]

        territory_start = 0 if pd.isna(prev_gene_end) else int(prev_gene_end)
        if pd.isna(next_gene_start):
            chrom_size = self.sequence_loader.chromsizes.get(chrom) if self.sequence_loader.chromsizes else None
            territory_end = chrom_size if chrom_size is not None else np.inf
        else:
            territory_end = int(next_gene_start)

        # 1. Compute the actual genomic window after shifting (same convention as `_get_shuffle_mask`)
        win_start = start + shift

        # 2. Build the mask over forward-strand genomic positions
        positions = win_start + np.arange(seq_len)
        mask = (positions >= territory_start) & (positions < territory_end)

        # 3. Mirror the mask to match the orientation of the returned sequence string
        if strand == "-":
            mask = mask[::-1]
        if revcomp:
            mask = mask[::-1]

        return mask

    def _encode_sequence(
        self,
        seq: str,
        original_index: str | None = None,
        parsed_index: tuple[str, int, int, str] | None = None,
        shift: int = 0,
        revcomp: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """One-hot encode `seq`, appending a 5th 'gene territory' channel if `gene_neighbors` was provided.

        `original_index`/`parsed_index` default to None to tolerate the input-shape probing call in
        `BaseGenomicDataWrapper.__init__`, which calls this method with only the sequence string.
        """
        x = one_hot_encode_sequence(seq, expand_dim=False)
        if self._gene_neighbors is None:
            return x
        if original_index is None or parsed_index is None:
            territory_channel = np.zeros((x.shape[0], 1), dtype=x.dtype)
        else:
            mask = self._get_gene_territory_mask(
                original_index=original_index, parsed_index=parsed_index, shift=shift, revcomp=revcomp,
            )
            territory_channel = mask.astype(x.dtype)[:, None]
        return np.concatenate([x, territory_channel], axis=-1)

    def get_config(self) -> dict:
        """Return a dict of properties, to be logged during training.

        Primarily used in Crested.fit().
        """
        config = super().get_config()
        config.update({
            "split_column": self.split_column,
            "compressed": self.compressed,
            "gene_neighbors_path": self.gene_neighbors_path,
            "gene_id_column": self.gene_id_column if self._gene_neighbors is not None else None,
        })
        return config
