"""Dataset class for combining genome files and AnnData objects."""

from __future__ import annotations

import re

import keras
import numpy as np
from anndata import AnnData
from loguru import logger
from scipy.sparse import spmatrix

from crested._genome import Genome
from crested.utils import one_hot_encode_sequence

from ._sequenceloader import SequenceLoader


def _flip_region_strand(region: str) -> str:
    """Reverse the strand of a region."""
    strand_reverser = {"+": "-", "-": "+"}
    return region[:-1] + strand_reverser[region[-1]]


def _check_strandedness(region: str) -> bool:
    """Check the strandedness of a region, raising an error if the formatting isn't recognised."""
    if re.fullmatch(r".+:\d+-\d+:[-+]", region):
        return True
    elif re.fullmatch(r".+:\d+-\d+", region):
        return False
    else:
        raise ValueError(
            f"Region {region} was not recognised as a valid coordinate set (chr:start-end or chr:start-end:strand)."
            "If provided, strand must be + or -."
        )


def _deterministic_shift_region(
    region: str, stride: int = 50, n_shifts: int = 2
) -> list[str]:
    """
    Shift each region by a deterministic stride to each side. Will increase the number of regions by n_shifts times two.

    This is a legacy function, it's recommended to use stochastic shifting instead.
    """
    new_regions = []
    chrom, start_end, strand = region.split(":")
    start, end = map(int, start_end.split("-"))
    for i in range(-n_shifts, n_shifts + 1):
        new_start = start + i * stride
        new_end = end + i * stride
        new_regions.append(f"{chrom}:{new_start}-{new_end}:{strand}")
    return new_regions


class IndexManager:
    """
    Manage indices for the dataset.

    Augments indices with strand information if always reverse complement.

    Parameters
    ----------
    indices
        List of indices in format "chr:start-end" or "chr:start-end:strand".
    always_reverse_complement
        If True, all sequences will be augmented with their reverse complement.
    deterministic_shift
        If True, each region will be shifted twice with stride 50bp to each side.
    """

    def __init__(
        self,
        indices: list[str],
        always_reverse_complement: bool,
        deterministic_shift: bool = False,
    ):
        """Initialize the IndexManager with the provided indices."""
        self.indices = indices
        self.always_reverse_complement = always_reverse_complement
        self.deterministic_shift = deterministic_shift
        self.augmented_indices, self.augmented_indices_map = self._augment_indices(
            indices
        )

    def shuffle_indices(self):
        """Shuffle indices. Managed by wrapping class AnnDataLoader."""
        np.random.shuffle(self.augmented_indices)

    def _augment_indices(self, indices: list[str]) -> tuple[list[str], dict[str, str]]:
        """Augment indices with strand information. Necessary if always reverse complement to map sequences back to targets."""
        augmented_indices = []
        augmented_indices_map = {}
        for region in indices:
            if not _check_strandedness(
                region
            ):  # If slow, can use AnnDataset stranded argument - but this validates every region's formatting as well
                stranded_region = f"{region}:+"
            else:
                stranded_region = region

            if self.deterministic_shift:
                shifted_regions = _deterministic_shift_region(stranded_region)
                for shifted_region in shifted_regions:
                    augmented_indices.append(shifted_region)
                    augmented_indices_map[shifted_region] = region
                    if self.always_reverse_complement:
                        augmented_indices.append(_flip_region_strand(shifted_region))
                        augmented_indices_map[_flip_region_strand(shifted_region)] = (
                            region
                        )
            else:
                augmented_indices.append(stranded_region)
                augmented_indices_map[stranded_region] = region
                if self.always_reverse_complement:
                    augmented_indices.append(_flip_region_strand(stranded_region))
                    augmented_indices_map[_flip_region_strand(stranded_region)] = region
        return augmented_indices, augmented_indices_map


if keras.config.backend() == "pytorch":
    import torch

    BaseClass = torch.utils.data.Dataset
else:
    BaseClass = object


class AnnDataset(BaseClass):
    """
    Dataset class for combining genome files and AnnData objects.

    Called by the by the AnnDataModule class.

    Parameters
    ----------
    anndata
        AnnData object containing the data.
    genome
        Genome instance
    split
        'train', 'val', or 'test' split column in anndata.var.
    in_memory
        If True, the train and val sequences will be loaded into memory.
    random_reverse_complement
        If True, the sequences will be randomly reverse complemented during training.
    always_reverse_complement
        If True, all sequences will be augmented with their reverse complement during training.
    max_stochastic_shift
        Maximum stochastic shift (n base pairs) to apply randomly to each sequence during training.
    deterministic_shift
        If true, each region will be shifted twice with stride 50bp to each side.
        This is our legacy shifting, we recommend using max_stochastic_shift instead.
    """

    def __init__(
        self,
        anndata: AnnData,
        genome: Genome,
        split: str = None,
        in_memory: bool = True,
        random_reverse_complement: bool = False,
        always_reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
        deterministic_shift: bool = False,
    ):
        """Initialize the dataset with the provided AnnData object and options."""
        self.anndata = self._split_anndata(anndata, split)
        self.split = split
        self.indices = list(self.anndata.var_names)
        self.in_memory = in_memory
        self.compressed = isinstance(self.anndata.X, spmatrix)
        self.index_map = {index: i for i, index in enumerate(self.indices)}
        self.num_outputs = self.anndata.X.shape[0]
        self.random_reverse_complement = random_reverse_complement
        self.max_stochastic_shift = max_stochastic_shift
        self.shuffle = False  # managed by wrapping class AnnDataLoader

        # Check region formatting
        stranded = _check_strandedness(self.indices[0])
        if stranded and (always_reverse_complement or random_reverse_complement):
            logger.info(
                "Setting always_reverse_complement=True or random_reverse_complement=True with stranded data.",
                "This means both strands are used when training and the strand information is effectively disregarded.",
            )

        self.index_manager = IndexManager(
            self.indices,
            always_reverse_complement=always_reverse_complement,
            deterministic_shift=deterministic_shift,
        )
        self.sequence_loader = SequenceLoader(
            genome,
            in_memory=in_memory,
            max_stochastic_shift=max_stochastic_shift,
            regions=self.index_manager.augmented_indices,
        )
        self.seq_len = len(
            self.sequence_loader.get_sequence(self.index_manager.augmented_indices[0], stranded=True)
        )

    @staticmethod
    def _split_anndata(anndata: AnnData, split: str) -> AnnData:
        """Return subset of anndata based on a given split column."""
        if split:
            if "split" not in anndata.var.columns:
                raise KeyError(
                    "No split column found in anndata.var. Run `pp.train_val_test_split` first."
                )
        subset = (
            anndata[:, anndata.var["split"] == split].copy()
            if split
            else anndata.copy()
        )
        return subset

    def __len__(self) -> int:
        """Get number of (augmented) samples in the dataset."""
        return len(self.index_manager.augmented_indices)

    def _get_target(self, index: str) -> np.ndarray:
        """Get target for a given index."""
        y_index = self.index_map[index]
        return (
            self.anndata.X[:, y_index].toarray().flatten()
            if self.compressed
            else self.anndata.X[:, y_index]
        )

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:
        """Return sequence and target for a given index."""
        augmented_index = self.index_manager.augmented_indices[idx]
        original_index = self.index_manager.augmented_indices_map[augmented_index]
        # stochastic shift
        if self.max_stochastic_shift > 0:
            shift = np.random.randint(
                -self.max_stochastic_shift, self.max_stochastic_shift + 1
            )
        else:
            shift = 0

        # Get sequence
        x = self.sequence_loader.get_sequence(
            augmented_index, stranded=True, shift=shift
        )

        # random reverse complement (always_reverse_complement is done in the sequence loader)
        if self.random_reverse_complement and np.random.rand() < 0.5:
            x = self.sequence_loader._reverse_complement(x)

        # one hot encode sequence and convert to numpy array
        x = one_hot_encode_sequence(x, expand_dim=False)
        y = self._get_target(original_index)

        return x, y

    def __call__(self):
        """Call generator for the dataset."""
        for i in range(len(self)):
            if i == 0:
                if self.shuffle:
                    self.index_manager.shuffle_indices()
            yield self.__getitem__(i)

    def __repr__(self) -> str:
        """Get string representation of the dataset."""
        return f"AnnDataset(anndata_shape={self.anndata.shape}, n_samples={len(self)}, num_outputs={self.num_outputs}, split={self.split}, in_memory={self.in_memory})"
