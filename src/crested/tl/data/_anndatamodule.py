"""Anndatamodule which acts as a wrapper around AnnDataset and AnnDataLoader."""

from __future__ import annotations

from os import PathLike

from crested._genome import Genome, _resolve_genome

from ._dataloader import AnnDataLoader
from ._dataset import AnnDataset


class AnnDataModule:
    """
    DataModule class which defines how dataloaders should be loaded in each stage.

    Required input for the `tl.Crested` class.

    Note
    ----
    Expects a `split` column in the `.var` DataFrame of the AnnData object.
    Run `pp.train_val_test_split` first to add the `split` column to the AnnData object if not yet done.

    Example
    -------
    >>> data_module = AnnDataModule(
    ...     adata,
    ...     genome=my_genome,
    ...     always_reverse_complement=True,
    ...     max_stochastic_shift=50,
    ...     batch_size=256,
    ... )

    Parameters
    ----------
    adata
        An instance of AnnData containing the data to be loaded.
    genome
        Instance of Genome or Path to the fasta file.
        If None, will look for a registered genome object.
    chromsizes_file
        Path to the chromsizes file. Not required if genome is a Genome object.
        If genome is a path and chromsizes is not provided, will deduce the chromsizes from the fasta file.
    in_memory
        If True, the train and val sequences will be loaded into memory. Default is True.
    always_reverse_complement
        If True, all sequences will be augmented with their reverse complement during training.
        Effectively increases the training dataset size by a factor of 2. Default is True.
    random_reverse_complement
        If True, the sequences will be randomly reverse complemented during training. Default is False.
    max_stochastic_shift
        Maximum stochastic shift (n base pairs) to apply randomly to each sequence during training. Default is 0.
    deterministic_shift
        If true, each region will be shifted twice with stride 50bp to each side. Default is False.
        This is our legacy shifting, we recommend using max_stochastic_shift instead.
    shuffle
        If True, the data will be shuffled at the end of each epoch during training. Default is True.
    batch_size
        Number of samples per batch to load. Default is 256.
    """

    def __init__(
        self,
        adata,
        genome: PathLike | Genome | None = None,
        chromsizes_file: PathLike | None = None,
        in_memory: bool = True,
        always_reverse_complement=True,
        random_reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
        deterministic_shift: bool = False,
        shuffle: bool = True,
        batch_size: int = 256,
    ):
        """Initialize the DataModule with the provided dataset and options."""
        self.adata = adata
        self.genome = _resolve_genome(genome, chromsizes_file)  # backward compatibility
        self.always_reverse_complement = always_reverse_complement
        self.in_memory = in_memory
        self.random_reverse_complement = random_reverse_complement
        self.max_stochastic_shift = max_stochastic_shift
        self.deterministic_shift = deterministic_shift
        self.shuffle = shuffle
        self.batch_size = batch_size

        self._validate_init_args(random_reverse_complement, always_reverse_complement)

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

    def setup(self, stage: str) -> None:
        """
        Set up the Anndatasets for a given stage.

        Generates the train, val, test or predict dataset based on the provided stage.
        Should always be called before accessing the dataloaders.
        Generally you don't need to call this directly, as this is called inside the `tl.Crested` trainer class.

        Parameters
        ----------
        stage
            Stage for which to setup the dataloader. Either 'fit', 'test' or 'predict'.
        """
        if stage == "fit":
            self.train_dataset = AnnDataset(
                self.adata,
                self.genome,
                split="train",
                in_memory=self.in_memory,
                always_reverse_complement=self.always_reverse_complement,
                random_reverse_complement=self.random_reverse_complement,
                max_stochastic_shift=self.max_stochastic_shift,
                deterministic_shift=self.deterministic_shift,
            )
            self.val_dataset = AnnDataset(
                self.adata,
                self.genome,
                split="val",
                in_memory=self.in_memory,
                always_reverse_complement=False,
                random_reverse_complement=False,
                max_stochastic_shift=0,
            )
        elif stage == "test":
            self.test_dataset = AnnDataset(
                self.adata,
                self.genome,
                split="test",
                in_memory=False,
                always_reverse_complement=False,
                random_reverse_complement=False,
                max_stochastic_shift=0,
            )
        elif stage == "predict":
            self.predict_dataset = AnnDataset(
                self.adata,
                self.genome,
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
        """:obj:`crested.tl.data.AnnDataLoader`: Training dataloader."""
        if self.train_dataset is None:
            raise ValueError("train_dataset is not set. Run setup('fit') first.")
        return AnnDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_remainder=False,
        )

    @property
    def val_dataloader(self):
        """:obj:`crested.tl.data.AnnDataLoader`: Validation dataloader."""
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
        """:obj:`crested.tl.data.AnnDataLoader`: Test dataloader."""
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
        """:obj:`crested.tl.data.AnnDataLoader`: Prediction dataloader."""
        if self.predict_dataset is None:
            raise ValueError("predict_dataset is not set. Run setup('predict') first.")
        return AnnDataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
        )

    def __repr__(self):
        """Return a string representation of the AnndataModule."""
        return (
            f"AnndataModule(adata={self.adata}, genome={self.genome}, "
            f"in_memory={self.in_memory}, "
            f"always_reverse_complement={self.always_reverse_complement}, "
            f"random_reverse_complement={self.random_reverse_complement}, "
            f"max_stochastic_shift={self.max_stochastic_shift}, shuffle={self.shuffle}, "
            f"batch_size={self.batch_size}"
        )
