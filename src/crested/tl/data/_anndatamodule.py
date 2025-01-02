"""Anndatamodule which acts as a wrapper around AnnDataset and AnnDataLoader."""

from __future__ import annotations

from os import PathLike
from torch.utils.data import Sampler
import numpy as np

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
        always_reverse_complement: bool = True,
        random_reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
        deterministic_shift: bool = False,
        shuffle: bool = True,
        batch_size: int = 256,
        obs_columns: list[str] | None = None,
        obsm_keys: list[str] | None = None,
        varp_keys: list[str] | None = None,
    ):
        """Initialize the DataModule with the provided dataset and options."""
        self.adata = adata
        self.genome = _resolve_genome(genome, chromsizes_file)  # Function assumed available
        self.always_reverse_complement = always_reverse_complement
        self.in_memory = in_memory
        self.random_reverse_complement = random_reverse_complement
        self.max_stochastic_shift = max_stochastic_shift
        self.deterministic_shift = deterministic_shift
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.obs_columns = obs_columns
        self.obsm_keys = obsm_keys
        self.varp_keys = varp_keys

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
        Generally, you don't need to call this directly, as this is called inside the `tl.Crested` trainer class.

        Parameters
        ----------
        stage
            Stage for which to setup the dataloader. Either 'fit', 'test' or 'predict'.
        """
        args = {
            "anndata": self.adata,
            "genome": self.genome,
            "in_memory": self.in_memory,
            "always_reverse_complement": self.always_reverse_complement,
            "random_reverse_complement": self.random_reverse_complement,
            "max_stochastic_shift": self.max_stochastic_shift,
            "deterministic_shift": self.deterministic_shift,
            "obs_columns": self.obs_columns,
            "obsm_keys": self.obsm_keys,
            "varp_keys": self.varp_keys,
        }
        if stage == "fit":
            # Training dataset
            train_args = args.copy()
            train_args["split"] = "train"

            val_args = args.copy()
            val_args["split"] = "val"
            val_args["always_reverse_complement"] = False
            val_args["random_reverse_complement"] = False
            val_args["max_stochastic_shift"] = 0

            self.train_dataset = AnnDataset(**train_args)
            self.val_dataset = AnnDataset(**val_args)

        elif stage == "test":
            test_args = args.copy()
            test_args["split"] = "test"
            test_args["in_memory"] = False
            test_args["always_reverse_complement"] = False
            test_args["random_reverse_complement"] = False
            test_args["max_stochastic_shift"] = 0

            self.test_dataset = AnnDataset(**test_args)

        elif stage == "predict":
            predict_args = args.copy()
            predict_args["split"] = None
            predict_args["in_memory"] = False
            predict_args["always_reverse_complement"] = False
            predict_args["random_reverse_complement"] = False
            predict_args["max_stochastic_shift"] = 0

            self.predict_dataset = AnnDataset(**predict_args)

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

class MetaAnnDataModule:
    """
    A DataModule for multiple AnnData objects (one per species),
    merging them into a single MetaAnnDataset for each stage. 
    Then we rely on the MetaSampler to do globally weighted sampling.

    This replaces the old random-per-dataset approach in MultiAnnDataset.
    """

    def __init__(
        self,
        adatas: list[AnnData],
        genomes: list[Genome],
        in_memory: bool = True,
        always_reverse_complement: bool = True,
        random_reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
        deterministic_shift: bool = False,
        shuffle: bool = True,
        batch_size: int = 256,
        obs_columns: list[str] | None = None,
        obsm_keys: list[str] | None = None,
        varp_keys: list[str] | None = None,
        epoch_size: int = 100_000,
    ):
        if len(adatas) != len(genomes):
            raise ValueError("Must provide as many `adatas` as `genomes`.")

        self.adatas = adatas
        self.genomes = genomes
        self.in_memory = in_memory
        self.always_reverse_complement = always_reverse_complement
        self.random_reverse_complement = random_reverse_complement
        self.max_stochastic_shift = max_stochastic_shift
        self.deterministic_shift = deterministic_shift
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.obs_columns = obs_columns
        self.obsm_keys = obsm_keys
        self.varp_keys = varp_keys
        self.epoch_size = epoch_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str) -> None:
        """
        Create the AnnDataset objects for each adata+genome, then unify them 
        into a MetaAnnDataset for the given stage.
        """
        def dataset_args(split):
            return {
                "in_memory": self.in_memory,
                "always_reverse_complement": self.always_reverse_complement,
                "random_reverse_complement": self.random_reverse_complement,
                "max_stochastic_shift": self.max_stochastic_shift,
                "deterministic_shift": self.deterministic_shift,
                "obs_columns": self.obs_columns,
                "obsm_keys": self.obsm_keys,
                "varp_keys": self.varp_keys,
                "split": split,
            }

        if stage == "fit":
            train_datasets = []
            val_datasets = []
            for adata, genome in zip(self.adatas, self.genomes):
                # Training
                args = dataset_args("train")
                ds_train = AnnDataset(anndata=adata, genome=genome, **args)
                train_datasets.append(ds_train)

                # Validation (no shifting, no RC)
                val_args = dataset_args("val")
                val_args["always_reverse_complement"] = False
                val_args["random_reverse_complement"] = False
                val_args["max_stochastic_shift"] = 0
                ds_val = AnnDataset(anndata=adata, genome=genome, **val_args)
                val_datasets.append(ds_val)

            # Merge them with MetaAnnDataset
            self.train_dataset = MetaAnnDataset(train_datasets)
            self.val_dataset = MetaAnnDataset(val_datasets)

        elif stage == "test":
            test_datasets = []
            for adata, genome in zip(self.adatas, self.genomes):
                args = dataset_args("test")
                args["in_memory"] = False
                args["always_reverse_complement"] = False
                args["random_reverse_complement"] = False
                args["max_stochastic_shift"] = 0

                ds_test = AnnDataset(anndata=adata, genome=genome, **args)
                test_datasets.append(ds_test)

            self.test_dataset = MetaAnnDataset(test_datasets)

        elif stage == "predict":
            predict_datasets = []
            for adata, genome in zip(self.adatas, self.genomes):
                args = dataset_args(None)
                args["in_memory"] = False
                args["always_reverse_complement"] = False
                args["random_reverse_complement"] = False
                args["max_stochastic_shift"] = 0

                ds_pred = AnnDataset(anndata=adata, genome=genome, **args)
                predict_datasets.append(ds_pred)

            self.predict_dataset = MetaAnnDataset(predict_datasets)

        else:
            raise ValueError(f"Invalid stage: {stage}")

    @property
    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset is not set. Run setup('fit') first.")
        return AnnDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_remainder=False,
            epoch_size=self.epoch_size,
        )

    @property
    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("val_dataset is not set. Run setup('fit') first.")
        return AnnDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
            epoch_size=self.epoch_size,
        )

    @property
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("test_dataset is not set. Run setup('test') first.")
        return AnnDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
            epoch_size=self.epoch_size,
        )

    @property
    def predict_dataloader(self):
        if self.predict_dataset is None:
            raise ValueError("predict_dataset is not set. Run setup('predict') first.")
        return AnnDataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_remainder=False,
            epoch_size=self.epoch_size,
        )

    def __repr__(self):
        return (
            f"MetaAnnDataModule("
            f"num_species={len(self.adatas)}, "
            f"batch_size={self.batch_size}, shuffle={self.shuffle}, "
            f"max_stochastic_shift={self.max_stochastic_shift}, "
            f"random_reverse_complement={self.random_reverse_complement}, "
            f"always_reverse_complement={self.always_reverse_complement}, "
            f"in_memory={self.in_memory}, "
            f"deterministic_shift={self.deterministic_shift}, "
            f"epoch_size={self.epoch_size}"
            f")"
        )
