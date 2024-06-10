"""Dataset class for combining genome files and AnnData objects."""

from __future__ import annotations

from os import PathLike

import numpy as np
from anndata import AnnData
from loguru import logger
from pysam import FastaFile
from scipy.sparse import spmatrix
from tqdm import tqdm


class AnnDataset:
    def __init__(
        self,
        anndata: AnnData,
        genome_file: PathLike,
        split: str = None,
        chromsizes_file: PathLike | None = None,
        in_memory: bool = True,
        random_reverse_complement: bool = False,
        always_reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
    ):
        self._validate_init_args(random_reverse_complement, always_reverse_complement)
        self.anndata = self._split_anndata(anndata, split)
        self.split = split
        self.indices = list(self.anndata.var_names)
        self.in_memory = in_memory
        self.compressed = isinstance(self.anndata.X, spmatrix)
        self.genome = FastaFile(genome_file)
        self.chromsizes = chromsizes_file
        self.index_map = {index: i for i, index in enumerate(self.indices)}
        self.shuffle = False
        self.num_outputs = self.anndata.X.shape[0]
        self.complement = str.maketrans("ACGT", "TGCA")
        self.random_reverse_complement = random_reverse_complement
        self.always_reverse_complement = always_reverse_complement

        if (chromsizes_file is None) and (max_stochastic_shift > 0):
            self._warn_no_chromsizes_file()

        if self.in_memory:
            self.sequences = self._load_sequences_into_memory()

        self.augmented_indices, self.augmented_indices_map = self._augment_indices(
            self.indices
        )

    @staticmethod
    def _validate_init_args(
        random_reverse_complement: bool, always_reverse_complement: bool
    ):
        if random_reverse_complement and always_reverse_complement:
            raise ValueError(
                "Only one of `random_reverse_complement` and `always_reverse_complement` can be True."
            )

    @staticmethod
    def _warn_no_chromsizes_file():
        logger.warning(
            "Chromsizes file not provided when shifting. Will not check if shifted regions are within chromosomes",
        )

    @staticmethod
    def _split_anndata(anndata: AnnData, split: str) -> AnnData:
        subset = (
            anndata[:, anndata.var["split"] == split].copy()
            if split
            else anndata.copy()
        )
        return subset

    def _load_sequences_into_memory(self) -> dict:
        logger.info("Loading sequences into memory...")
        sequences = {}
        for region in tqdm(self.indices):
            sequences[f"{region}:+"] = self._get_sequence(region)
            if self.always_reverse_complement:
                sequences[f"{region}:-"] = self._reverse_complement(
                    sequences[f"{region}:+"]
                )
        return sequences

    def _augment_indices(self, indices: list[str]) -> tuple[list[str], dict[str, str]]:
        augmented_indices = []
        augmented_indices_map = {}
        for region in indices:
            augmented_indices.append(f"{region}:+")
            augmented_indices_map[f"{region}:+"] = region
            if self.always_reverse_complement:
                augmented_indices.append(f"{region}:-")
                augmented_indices_map[f"{region}:-"] = region
        return augmented_indices, augmented_indices_map

    def __len__(self) -> int:
        return len(self.augmented_indices)

    def _get_sequence(self, region: str) -> str:
        chrom, start_end = region.split(":")
        start, end = start_end.split("-")
        return self.genome.fetch(chrom, int(start), int(end))

    def _reverse_complement(self, sequence: str) -> str:
        return sequence.translate(self.complement)[::-1]

    def _get_target(self, index: str) -> np.ndarray:
        y_index = self.index_map[index]
        return (
            self.anndata.X[:, y_index].toarray().flatten()
            if self.compressed
            else self.anndata.X[:, y_index]
        )

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:
        augmented_index = self.augmented_indices[idx]
        original_index = self.augmented_indices_map[augmented_index]

        x = (
            self.sequences[augmented_index]
            if self.in_memory
            else self._get_sequence(original_index)
        )
        if augmented_index.endswith(":-"):
            x = self._reverse_complement(x)
        if self.random_reverse_complement and np.random.rand() < 0.5:
            x = self._reverse_complement(x)

        y = self._get_target(original_index)
        return x, y

    def __call__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

        if i == (len(self) - 1):
            if self.shuffle:
                self._shuffle_indices()

    def _shuffle_indices(self):
        np.random.shuffle(self.indices)
        self.augmented_indices, self.augmented_indices_map = self._augment_indices(
            self.indices
        )

    def __repr__(self) -> str:
        return f"AnnDataset(anndata_shape={self.anndata.shape}, n_samples={len(self)}, num_outputs={self.num_outputs}, split={self.split}, in_memory={self.in_memory})"
