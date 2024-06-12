"""Anndatamodule which acts as a wrapper around AnnDataset and AnnDataLoader."""

from __future__ import annotations

from os import PathLike

from loguru import logger

from ._dataloader import AnnDataLoader
from ._dataset import AnnDataset


class AnnDataModule:
    """DataModule class which defines how dataloaders should be loaded in each stage."""

    def __init__(
        self,
        adata,
        genome_file,
        chromsizes_file: PathLike | None = None,
        in_memory: bool = True,
        always_reverse_complement=True,
        random_reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
        shuffle: bool = True,
        batch_size: int = 256,
        drop_remainder: bool = True,
    ):
        """
        Initialize the DataModule with the provided dataset and options.

        Parameters
        ----------
        adata
            An instance of AnnData containing the data to be loaded.
        genome_file
            Path to the genome file.
        chromsizes_file
            Path to the chromsizes file. Advised if max_stochastic_shift > 0.
        in_memory
            If True, the train and val sequences will be loaded into memory. Default is True.
        always_reverse_complement
            If True, all sequences will be augmented with their reverse complement during training.
            Effectively increases the training dataset size by a factor of 2. Default is True.
        random_reverse_complement
            If True, the sequences will be randomly reverse complemented during training. Default is False.
        max_stochastic_shift
            Maximum stochastic shift (n base pairs) to apply randomly to each sequence during training. Default is 0.
        shuffle
            If True, the data will be shuffled at the end of each epoch during training. Default is True.
        batch_size
            Number of samples per batch to load. Default is 256.
        drop_remainder
            If True, the last batch will be dropped if it is smaller than batch_size during training. Default is True.
        """
        self.adata = adata
        self.genome_file = genome_file
        self.chromsizes_file = chromsizes_file
        self.always_reverse_complement = always_reverse_complement
        self.in_memory = in_memory
        self.random_reverse_complement = random_reverse_complement
        self.max_stochastic_shift = max_stochastic_shift
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder

        self._validate_init_args(random_reverse_complement, always_reverse_complement)
        if (chromsizes_file is None) and (max_stochastic_shift > 0):
            self._warn_no_chromsizes_file()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

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

    def setup(self, stage: str):
        """Setup the Anndatasets for a given stage."""
        if stage == "fit":
            self.train_dataset = AnnDataset(
                self.adata,
                self.genome_file,
                split="train",
                chromsizes_file=self.chromsizes_file,
                in_memory=self.in_memory,
                always_reverse_complement=self.always_reverse_complement,
                random_reverse_complement=self.random_reverse_complement,
                max_stochastic_shift=self.max_stochastic_shift,
            )
            self.val_dataset = AnnDataset(
                self.adata,
                self.genome_file,
                split="val",
                in_memory=self.in_memory,
                always_reverse_complement=False,
                random_reverse_complement=False,
                max_stochastic_shift=0,
            )
        elif stage == "test":
            self.test_dataset = AnnDataset(
                self.adata,
                self.genome_file,
                split="test",
                in_memory=False,
                always_reverse_complement=False,
                random_reverse_complement=False,
                max_stochastic_shift=0,
            )
        elif stage == "predict":
            self.predict_dataset = AnnDataset(
                self.adata,
                self.genome_file,
                split=None,
                in_memory=False,
                always_reverse_complement=False,
                random_reverse_complement=False,
                max_stochastic_shift=0,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")

    @property
    def train_dataloader(self):
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("train_dataset is not set. Run setup('fit') first.")
        return AnnDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_remainder=self.drop_remainder,
        )

    @property
    def val_dataloader(self):
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("val_dataset is not set. Run setup('fit') first.")
        return AnnDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
        )

    @property
    def test_dataloader(self):
        """Return the test dataloader."""
        if self.test_dataset is None:
            raise ValueError("test_dataset is not set. Run setup('test') first.")
        return AnnDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
        )

    @property
    def predict_dataloader(self):
        """Return the prediction dataloader."""
        if self.predict_dataset is None:
            raise ValueError("predict_dataset is not set. Run setup('predict') first.")
        return AnnDataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
        )

    def __repr__(self):
        return (
            f"AnndataModule(adata={self.adata}, genome_file={self.genome_file}, "
            f"chromsizes_file={self.chromsizes_file}, in_memory={self.in_memory}, "
            f"always_reverse_complement={self.always_reverse_complement}, "
            f"random_reverse_complement={self.random_reverse_complement}, "
            f"max_stochastic_shift={self.max_stochastic_shift}, shuffle={self.shuffle}, "
            f"batch_size={self.batch_size}, drop_remainder={self.drop_remainder})"
        )
