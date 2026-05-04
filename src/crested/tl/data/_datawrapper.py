"""DataWrapper class to load and transform model inputs and outputs from your object of choice."""

from __future__ import annotations

from math import ceil, inf

import keras
import numpy as np
from loguru import logger

if keras.config.backend() == "torch":
    import torch
    from torch.utils.data import DataLoader, default_collate
    FrameworkDatasetClass = torch.utils.data.Dataset
else:
    import tensorflow as tf
    FrameworkDatasetClass = object

from crested._genome import Genome, _resolve_genome
from crested.utils import flip_region_strand, one_hot_encode_sequence, parse_region

from ._sequenceloader import SequenceLoader


class BaseDataWrapper:
    """
    Class for retrieving sequences and their associated target values.

    Please inherit this and implement self._get_sequence(), self._get_target(), self._get_indices() and self._get_splits().
    If training on sequences extracted from the genome, have a look at BaseGenomicDataWrapper for built-in sequence loading.

    Parameters
    ----------
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

    Note
    ----
    A note on index types:
    - 'index'/'original_index': an identifier of a sample from the input data, before augmentation or anything.
        Think .var_names from your adata (whether that's regions or gene names).
        Returned by `_get_indices` in your inheriting function, and should have matching split values from `_get_splits`.
    - 'expanded_index': an identifier after expanding the indices. This often means adding the reverse complement, but it can be more if you implement it.
        Multiple expanded indices can map back to one index (i.e. forward and backward sequence map back to the same originating sequence and its linked value in the anndata).
        Note that due to TensorFlow limitations, this should be something convertable to numpy values and still usable as a dictionary key, so e.g. a string or an integer but not a tuple.
        If you don't have any use for expansion, you can change _expand_indices to simply return a 1-to-1 {index: index} mapping.
    - 'parsed_index': an expanded index parsed to a non-string/integer format, if useful.
        In the genomic context this is often from a string to a (chrom, start, end, strand) tuple, to prevent having to parse the region multiple times.
        If you don't have any use for parsing, you can keep `_parse_index` to just return the index as is (which is the default for BaseDataWrapper, but not BaseGenomicDataWrapper).
    """

    # ----- Object initialization -----
    def __init__(
        self,
        batch_size: int = 256,
        random_reverse_complement: bool = False,
        always_reverse_complement: bool = True,
        max_stochastic_shift: int = 0,
        drop_remainder: bool = False,
        train_splits: str | list | None = None,
        val_splits: str | list = 'val',
        test_splits: str | list = 'test',
    ):
        """Initialize the DataWrapper with the provided dataset and options."""
        # Dataset
        # Split parameters
        if isinstance(train_splits, str):
            train_splits = [train_splits]
        if isinstance(val_splits, str):
            val_splits = [val_splits]
        if isinstance(test_splits, str):
            test_splits = [test_splits]
        if train_splits is None:
            train_splits = list(set(self._get_splits()) - (set(val_splits) | set(test_splits)))
            logger.info(f"Training split labels inferred to be {train_splits}.")
        self.split_labels = {'train': train_splits, 'val': val_splits, 'test': test_splits}

        # Data augmentation parameters
        self.random_reverse_complement = random_reverse_complement
        self.always_reverse_complement = always_reverse_complement
        self.max_stochastic_shift = max_stochastic_shift
        # DataLoader parameters
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        # Check revcomp parameters
        if random_reverse_complement and always_reverse_complement:
            raise ValueError(
                "Only one of `random_reverse_complement` and `always_reverse_complement` can be True."
            )

        # Infer in inherited class, or leave input_shape()/output_shape() to infer them later
        self.input_shape_cache = None
        self.output_shape_cache = None

        # Get device if using torch
        if keras.config.backend() == "torch":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Process indices
        self._process_indices()

    def _process_indices(self):
        """Get, split, expand and save indices, using _get_indices(), _split_indices() and _expand_indices()."""
        self.indices = self._get_indices()
        self.split_indices = {split: self._split_indices(split) for split in  ('train', 'val', 'test')}
        # Check if indices (look like they) contain stranded information
        # Note: don't use _check_region_strandedness since that also checks for region formatting, while we want to allow all kinds of indices here
        if isinstance(self.indices[0], str) and self.indices[0][-1] in ('+', '-'):
            self.stranded = True
        else:
            self.stranded = False

        # Set up sequence index handling to match deterministic augmentation (always_reverse_complement) to original output values
        # Pass all values through _expand_indices (since it makes sure all values are stranded), for training also actually add the reverse complement version (if always_rev_comp=True)

        # Get stranded indices for entire dataset (for prediction)
        self.split_expanded_indices = {}
        self.split_expanded_indices['predict'] = self._expand_indices(self.indices, expand_revcomp = False)
        # Get stranded indices per split, expanding with revcomp for training set
        self.split_expanded_indices['train'] = self._expand_indices(self.split_indices['train'], expand_revcomp = self.always_reverse_complement)
        for split in ('val', 'test'):
            self.split_expanded_indices[split] = self._expand_indices(self.split_indices[split], expand_revcomp = False)
        # Get total set of expanded indices across all possibilities
        self.full_expanded_indices = list({aug_index for aug_index_list in self.split_expanded_indices.values() for aug_index in aug_index_list})
        # Get forward mapping (non-expanded to expanded) from predict
        self.expanded_indices_forward_map = dict(zip(self.indices, self.split_expanded_indices['predict'], strict=True))
        # Get mapping of expanded indices to parsed indices
        self.parsed_indices_map = {expanded_index: self._parse_index(expanded_index) for expanded_index in self.full_expanded_indices}

    # ----- Primary methods to implement/overwrite in the inherited versions: -----
    def _get_indices(self):
        """Get a list of all indices. Indices (possibly after expansion) are used in _get_sequence() and _get_target() to retrieve sequence and targets for that entry, respectively."""
        # Pseudocode: return self.data.metadata.index
        raise NotImplementedError("Please define `self._get_indices()` to extract some kind of unique ID per sample from your dataset, i.e. AnnData var_names or DataFrame index.")

    def _get_splits(self):
        """Get values that map to train/val/test splits for each index.

        Expected to return values according to train/val/test_splits, so if you have train_splits='fold0', val_splits='fold1', test_splits='fold2',
        _get_splits should return a list of values in the set {'fold0', 'fold1', 'fold2'}.
        """
        # Pseudocode: return self.data.metadata[self.split_column]
        raise NotImplementedError("Please define `self._get_splits()` to extract a split identifier per sample from your dataset, i.e. AnnData var['split'].")

    def _get_sequence(
        self,
        original_index: str,
        expanded_index: str,
        parsed_index: tuple[str, int, int, str],
        revcomp: bool = False,
        shift: int = 0,
        **kwargs
    ) -> str:
        """Retrieve the sequence (as a string) of a certain index.

        Use the parameters you need and disregard the rest through **kwargs.

        Parameters
        ----------
        original_index
            The original index of the sequence (i.e. before always_reverse_complement or other expansion).
        expanded_index
            The expanded index of the sequence (guaranteed to be stranded).
        parsed_index
            The expanded index of the sequence, parsed into a (chrom, start, end, strand) tuple if using genomic data.
        revcomp
            Whether to reverse-complement the string (like because of stochastic reverse complementing) relative to the requested original index.
        shift
            How much to shift the string left or right (like because of stochastic shifting) relative to the original requested index.
            Can be positive or negative.
        kwargs
            Catcher for arguments you're not using in your implementation.
        """
        raise NotImplementedError("Implement _get_sequence of your inherited DataWrapper class to retrieve (unencoded) input sequences for your classes, to be encoded by _encode_sequence.")

    def _encode_sequence(self, seq: str) -> np.ndarray:
        """Encode the sequence as string to a numerical representation by one-hot encoding it. Returned value should not have a batch dimension yet.

        Generally done through one_hot_encode_sequence, override if you need to do something fancier.

        Parameters
        ----------
        seq
            A sequence as a string, to encode as a one-hot encoded numpy array.
        """
        return one_hot_encode_sequence(seq, expand_dim=False)

    def _get_target(
        self,
        original_index: str,
        expanded_index: str,
        parsed_index: tuple[str, int, int, str],
        revcomp: bool = False,
        shift: int = 0,
        **kwargs
    ) -> np.ndarray:
        """Get target for a given index. Returned value should not have a batch dimension yet.

        If not using certain arguments in your implementation (like only using one of original_index/expanded_index), please keep **kwargs to absorb the un-used other arguments.

        Parameters
        ----------
        original_index
            The original index of the sequence (i.e. before always_reverse_complement or other expansion).
            The inherited version will generally use only original_index, expanded_index or parsed_index, depending on desired functionality.
        expanded_index
            The expanded index of the sequence (guaranteed to be stranded).
            The inherited version will generally use only original_index, expanded_index or parsed_index, depending on desired functionality.
        parsed_index
            The expanded index, parsed into a more usable format. By default, into a (chrom, start, end, strand) tuple if using genomic data.
            The inherited version will generally use only original_index, expanded_index or parsed_index, depending on desired functionality.
        revcomp
            Whether to reverse-complement the string (like because of stochastic reverse complementing) relative to the requested index.
        shift
            How much to shift the string left or right (like because of stochastic shifting) relative to the requested index.
            Can be positive or negative.
        kwargs
            Catcher for arguments you're not using in your implementation.
        """
        raise NotImplementedError("Implement _get_target() of your inherited DataWrapper class to retrieve output values for your sequence")

    def _get_shift(self, **kwargs) -> int:
        """Return a stochastic shift value.

        If there are constraints to your stochastic shift, like genomic limits, this function can be overwritten.
        """
        rng = np.random.default_rng()
        return rng.integers(
            -self.max_stochastic_shift, self.max_stochastic_shift, endpoint=True
        )

    def _get_expanded_index(self, index):
        """Get an expanded index from on the original index.

        By default, assumes the index is a region string and adds a strand if not provided.
        If using another way to get transformed indices, like looking up in .var columns, overwrite this.
        If not using transformed indices, simply pass the original index back.
        """
        if not self.stranded:
            stranded_region = f"{index}:+"
        else:
            stranded_region = index
        return stranded_region

    def _parse_index(self, index):
        """Parse an index into an easier-to-use format.

        In certain setups, like when loading sequences on the fly or when extracting values from tracks, it pays to parse the expanded index
        (which must be a string or integer or the like, to to allow using as dictionary keys even when transformed to numpy) only once.
        In those cases, this function can be inherited to split them, which results in parsed_index for the sequence/target/etc data.
        See for example BaseGenomicDataWrapper, which parses and unparses between string and tuple versions of genomic regions.
        """
        return index

    def _unparse_index(self, index):
        """
        Turn a parsed index back into an indexable state (str, int, etc) from the parsed state.

        This can be inherited to unparse parsed indices back into their original form, if useful. See BaseGenomicDataWrapper for an example.
        If not inherited, parsed indices will be equivalent to unparsed.
        """
        return index

    # ----- Index management -----
    def _split_indices(self, split: str) -> list[str]:
        """Split the list of indices according to the split they belong to."""
        # Check whether train/val/test values are in the split column
        for split_label in self.split_labels[split]:
            if split_label not in self._get_splits():
                raise ValueError(f"Could not find {split} split value {split_label} in your split data. Split data example: {self._get_splits()[:5]}")
        # Get list of indices that have a split label in the split_labels for that split (i.e. 'fold0' for split 'train')
        return [index for index, index_split in zip(self.indices, self._get_splits(), strict=True) if index_split in self.split_labels[split]]

    def _expand_indices(self, indices: list[str], expand_revcomp: bool) -> list[str]:
        """Add strand information to indices, if not already present. Optionally also expands total set of indices by adding the reverse complement version index.

        Parameters
        ----------
        indices
            List of string-based indices to use.
        expand_revcomp
            Whether to expand the dataset by including both strands of every input index.
        """
        if not hasattr(self, 'expanded_indices_map'):
            self.expanded_indices_map = {}
        expanded_indices = []
        for index in indices:
            expanded_index = self._get_expanded_index(index) # also adds strand if not included
            expanded_indices.append(expanded_index)
            self.expanded_indices_map[expanded_index] = index # Update shared backmapping with the original indices
            if expand_revcomp:
                expanded_indices.append(flip_region_strand(expanded_index))
                self.expanded_indices_map[flip_region_strand(expanded_index)] = index
        return expanded_indices


    # ----- Item retrieval -----
    def __getitem__(self, idx: int | str) -> tuple[np.ndarray, np.ndarray]:
        """
        Return sequence and target for a given numeric or expanded str index. Not used too much: primary indexing function is get_indexed_item().

        Main logic is implemented in get_indexed_item() to retrieve using expanded indices and with optional stochastic augmentation.

        Parameters
        ----------
        idx
            The index to retrieve. If a string, used directly in self.get_indexed_item.
            If an integer, the associated index from the 'predict' expanded indices is used, just to have an easy way to get an example value.
        """
        if isinstance(idx, int):
            # Get expanded index (after adding revcomp indices) from prediction list
            # Generally a guaranteed-stranded version of the original index.
            return self.get_indexed_item(expanded_index=self.split_expanded_indices['predict'][idx])
        else:
            return self.get_indexed_item(expanded_index=idx)

    def get_indexed_item(self, expanded_index: str | None = None, original_index: str | None = None, augment: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve sequence and target for an expanded index.

        Parameters
        ----------
        expanded_index
            The main way to index into the object. An expanded index value, generally the index with a strand encoding.
            If kept as None, original_index is required
        original_index
            Generally optional. If your _get_sequence and _get_target ignore expanded_index, this is a way to use the original index directly,
            keeping expanded_index as None.
            If None (default), retrieves original version from expanded_index.
        augment
            Whether to add stochastic augmentation when retrieving items. Default is False.

        Returns
        -------
        Tuple of (sequence, target) as two numpy arrays.
        """
        if expanded_index is None and original_index is None:
            raise ValueError("Need to provide one of expanded_index or original_index to retrieve items.")
        elif original_index is None:
            original_index = self.expanded_indices_map[expanded_index]
        elif expanded_index is None:
            expanded_index = self.expanded_indices_forward_map[original_index]

        parsed_index = self.parsed_indices_map[expanded_index]

        # Get stochastic shift amount
        if augment and self.max_stochastic_shift > 0:
            shift = self._get_shift(original_index=original_index, expanded_index=expanded_index, parsed_index=parsed_index)
        else:
            shift = 0
        # Get whether to random reverse complement (always_reverse_complement is done in the sequence loader by encoding the strand in the index)
        if augment and self.random_reverse_complement and np.random.rand() < 0.5:
            revcomp = True
        else:
            revcomp = False
        # Get sequence as string and apply stochastic shift/revcomp augmentations, letting the function choose
        # whether to use original (like when grabbing from dataframe) or expanded index (like when extracting from genome)
        x = self._get_sequence(original_index=original_index, expanded_index=expanded_index, parsed_index=parsed_index, revcomp=revcomp, shift=shift)

        # Encode sequence (one-hot by default, but can override for i.e. tokenisation)
        x = self._encode_sequence(x)

        # Get targets, letting the function choose whether to use original (like with scalars) or expanded index (like with tracks)
        y = self._get_target(original_index=original_index, expanded_index=expanded_index, parsed_index=parsed_index, revcomp=revcomp, shift=shift)

        return x, y

    # ----- Item batching (PyTorch) -----
    def _collate_fn(self, batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[keras.KerasTensor], list[keras.KerasTensor]]:
        """Collate function to combine entries into a batch and move tensors to the specified device if backend is torch.

        Should work in most cases - in case you need to add batch-specific padding or something else, overwrite this function.
        """
        # default_collate is recursive (can also handle tuples of arrays), automatically converts to tensor
        batch = default_collate(batch)
        batch = recursive_move_device(batch, self.device)
        return batch

    # ----- Dataloader creation -----
    def create_dataloader(self, split: str | None, augment: bool = False, shuffle: bool = False):
        """Create a dataloader class that loops and batches the indices of choice, for use with model.fit() or your backend training loop of choice.

        If using the PyTorch backend, this will return a DataLoader. If using the TensorFlow backend, this will return a Dataset.
        Useful defaults are accessible as DataWrapper.[train/val/test/predict]_dataloader.

        Parameters
        ----------
        split
            str or None, indicating the split indices to loop over. Must be one of 'train', 'val', 'test' or 'predict'. None is a shorthand for 'predict'.
        augment
            Whether to use stochastic augmentation when retrieving sequences and targets.
        shuffle
            Whether to shuffle the indices.

        Returns
        -------
        A configured tf.data.Dataset or torch.utils.data.DataLoader object, iterating over the split of choice.
        """
        # PyTorch loops based on __len__ and __getitem__
        if keras.config.backend() == "torch":
            return DataLoader(
                self._create_looper(split=split, augment=augment),
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=self.drop_remainder,
                num_workers=0, # TODO: investigate whether we should adjust this to some autotune amount
                collate_fn=self._collate_fn,
            )
        # TensorFlow loops using a generator
        elif keras.config.backend() == "tensorflow":
            ds = tf.data.Dataset.from_generator(
                self._create_looper(split=split, augment=augment, shuffle_generator=shuffle).get_generator,
                output_signature=recursive_tensor_spec(self[0]),
            )
            ds = (
                ds.batch(self.batch_size, drop_remainder=self.drop_remainder)
                .repeat()
                .prefetch(tf.data.AUTOTUNE)
            )
            return ds
        else:
            return ImportError(f"CREsted currently only supports backends 'tensorflow' and 'torch', not backend {keras.config.backend()}")

    def _create_looper(self, split: str, augment: bool = False, shuffle_generator: bool = False) -> DataLooper:
        """Create a DataLooper mini-wrapper around the DataWrapper, using the indices of the split requested."""
        if split is None:
            split = 'predict'
        if split not in self.split_expanded_indices:
            raise ValueError(f"split {split} must be in self.split_expanded_indices. Available splits: {list(self.split_expanded_indices.keys())}")
        return DataLooper(datawrapper=self, indices=self.split_expanded_indices[split], augment=augment, shuffle_generator=shuffle_generator)

    @property
    def train_dataloader(self):
        """Return a tf.data.Dataset or torch.utils.data.DataLoader instance of the training samples, augmented."""
        return self.create_dataloader('train', augment=True, shuffle=True)
    @property
    def val_dataloader(self):
        """Return a tf.data.Dataset or torch.utils.data.DataLoader instance of the validation samples."""
        return self.create_dataloader('val', augment=False, shuffle=False)
    @property
    def test_dataloader(self):
        """Return a tf.data.Dataset or torch.utils.data.DataLoader instance of the test samples.."""
        return self.create_dataloader('test', augment=False, shuffle=False)
    @property
    def predict_dataloader(self):
        """Return a tf.data.Dataset or torch.utils.data.DataLoader instance of all samples."""
        return self.create_dataloader('predict', augment=False, shuffle=False)

    # ----- Dataset properties -----
    def get_config(self) -> dict:
        """Return a dict of properties, to be logged during training.

        Primarily used in Crested.fit().
        """
        return {
            'batch_size': self.batch_size,
            'drop_remainder': self.drop_remainder,
            'n_train_steps_per_epoch': self.batched_length('train'),
            'n_val_steps_per_epoch': self.batched_length('val'),
            'n_test_steps_per_epoch': self.batched_length('test'),
            'n_predict_steps_per_epoch': self.batched_length('predict'),
            'n_train': self.split_len('train'),
            'n_val': self.split_len('val'),
            'n_test': self.split_len('test'),
            'n_predict': self.split_len('predict'),
            'dataset_size': len(self),
            'seq_len': self.input_shape[-2],
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            "random_reverse_complement": self.random_reverse_complement,
            "always_reverse_complement": self.always_reverse_complement,
            "max_stochastic_shift": self.max_stochastic_shift,
            "train_splits": self.split_labels['train'],
            "val_splits": self.split_labels['val'],
            "test_splits": self.split_labels['test'],
        }

    def batched_length(self, split = None) -> int:
        """Return the number of batches in the DataLoader based on the dataset size and batch size."""
        if split is None:
            dataset_len = len(self)
        else:
            dataset_len = self.split_len(split)
        if self.drop_remainder:
            return dataset_len // self.batch_size
        else:
            return ceil(dataset_len/self.batch_size)

    def split_len(self, split: str) -> int:
        """Get length (including expanded revcomp samples) of the requested split."""
        return len(self.split_expanded_indices[split])

    @property
    def input_shape(self):
        """Shape of a single returned input, generally a sequence."""
        if self.input_shape_cache is None:
            self.input_shape_cache = recursive_shape(self[0][0])
        return self.input_shape_cache

    @property
    def output_shape(self):
        """Shape of a single returned output."""
        if self.output_shape_cache is None:
            self.output_shape_cache = recursive_shape(self[0][1])
        return self.output_shape_cache

    def __len__(self) -> int:
        """Get number of samples in the datawrapper, when looping in 'predict' mode."""
        return len(self.split_expanded_indices['predict'])

    def __repr__(self) -> str:
        """Get string representation of the wrapped dataset."""
        return (
            f"{self.__class__.__name__}: n_samples={len(self)}, (train: {self.split_len('train')}, val: {self.split_len('val')}, test: {self.split_len('test')}), "
            f"batch_size={self.batch_size}, input_shape={self.input_shape}, output_shape={self.output_shape}",
        )
             # TODO: expand (without self.data? or just leave to inheriting classes to figure out)


class DataLooper(FrameworkDatasetClass):
    """A small wrapper around a DataWrapper class, using (a subset of) indices and providing integer indexing (for PyTorch) and a generator (for TensorFlow).

    Meant to be used within tf.data.Dataset.from_generator() or torch's DataLoader (this subclasses torch.utils.data.Dataset).

    Parameters
    ----------
    datawrapper
        The DataWrapper object to retrieve samples from.
    indices
        The list of indices, which should work with datawrapper.get_indexed_item().
    augment
        Whether to use stochastic augmentation, passed to datawrapper.get_indexed_item().
    shuffle_generator
        Whether to shuffle the generator (used in TensorFlow). In PyTorch, please use DataLoader(DataLooper, shuffle=True) instead.
    """

    def __init__(
        self,
        datawrapper,
        indices: list[str],
        augment: bool,
        shuffle_generator: bool = False
    ):
        """Initialize the DataLooper class."""
        self.datawrapper = datawrapper
        self.indices = indices
        self.augment = augment
        self.shuffle_generator = shuffle_generator


    # ----- PyTorch indexing side -----
    def __getitem__(self, idx: int):
        """Retrieve the (input, output) pair for the region index with integer index idx."""
        return self.datawrapper.get_indexed_item(self.indices[idx], augment=self.augment)

    def __len__(self):
        """Return the number of indices."""
        return len(self.indices)

    # ----- TensorFlow generator side -----
    def get_generator(self):
        """Create a generator looping over the indices, shuffling if self.shuffle_generator = True."""
        loop_indices = self.indices
        if self.shuffle_generator:
            rng = np.random.default_rng()
            loop_indices = rng.permutation(loop_indices, axis = 0)
        for index in loop_indices:
            yield self.datawrapper.get_indexed_item(index, augment=self.augment)

class BaseGenomicDataWrapper(BaseDataWrapper):
    """
    Version of BaseDataWrapper with genomic sequence loading built-in.

    Please inherit this and implement self._get_target(), self._get_indices() and self._get_splits().

    Parameters
    ----------
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
    in_memory
        If True, extract the sequences from the genome before training starts and save them to memory,
        rather than extracting on the fly every time. Default is True.
    drop_remainder
        If True, drop the last batch if it is not the full batch_size. Default is False.
    train_splits
        The values in your split labeling that correspond to the training set as string or list of strings, i.e 'train' or ['fold0', 'fold1', 'fold2']
        If None, uses the values that aren't `val_splits` or `test_splits`.
    val_splits
        The values in your split labeling that correspond to the validation set as string or list of strings, i.e 'val' or ['fold3', 'fold4']
    test_splits
        The values in your split labeling that correspond to the test set as string or list of strings, i.e 'test' or ['fold5', 'fold6']
    kwargs
        Remaining keyword arguments, passed to BaseDataLoader.
    """

    def __init__(
        self,
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
        **kwargs
    ):
        """Initialize the DataWrapper with the provided dataset, a genome, and various options."""
        super().__init__(
            batch_size=batch_size,
            random_reverse_complement=random_reverse_complement,
            always_reverse_complement=always_reverse_complement,
            max_stochastic_shift=max_stochastic_shift,
            drop_remainder=drop_remainder,
            train_splits=train_splits,
            val_splits=val_splits,
            test_splits=test_splits,
            **kwargs
        )
        genome =  _resolve_genome(genome)

        self.sequence_loader = SequenceLoader(
            genome,
            in_memory=in_memory,
            max_stochastic_shift=self.max_stochastic_shift,
            regions=list(self.parsed_indices_map.values()),
        )

        # Save input shape now that we've created the function to retrieve one
        # Also pass original_index just in case we override _get_sequence in an inheriting class and use it
        example_index = self.split_expanded_indices['predict'][0]
        self.input_shape_cache = self._encode_sequence(self._get_sequence(
            parsed_index=self.parsed_indices_map[example_index],
            expanded_index=example_index,
            original_index=self.expanded_indices_map[example_index],
        )).shape

        if self.input_shape_cache[-1] > self.input_shape_cache[-2]:
            logger.warning(
                "Keras (and therefore CREsted) expects sequences to be encoded as (seq_len, n_nucleotides)."
                f"Your input shape appears to be flipped: {self.input_shape_cache}"
            )

    def _get_sequence(self, parsed_index: str, revcomp: bool = False, shift: int = 0, **kwargs) -> np.ndarray:
        """Get a sequence (as a string) given a certain index.

        Parameters
        ----------
        parsed_index
            The parsed index of the sequence (a (chrom, start, end, str) tuple).
        revcomp
            Whether to reverse-complement the string (like because of stochastic reverse complementing) relative to the requested index.
        shift
            How much to shift the string left or right (like because of stochastic shifting) relative to the requested index.
            Can be positive or negative.
        kwargs
            Catcher for unused arguments from `get_indexed_item()`, specifically `original_index` and `parsed_index`.
        """
        # We need the stranded information when extracting from the genome so use the expanded index
        x = self.sequence_loader.get_sequence(
            parsed_index, stranded=True, shift=shift
        )
        if revcomp:
            x = self.sequence_loader._reverse_complement(x)
        return x

    def _get_shift(self, parsed_index: [str, int, int, str], **kwargs) -> int:
        """Return a stochastic shift value, accounting for genomic limits given the index to shift.

        Parameters
        ----------
        parsed_index
            The parsed index of the sequence (a (chrom, start, end, strand) tuple).
        kwargs
            Catcher for unused arguments from `get_indexed_item()`, specifically `original_index` and `expanded_index`.
        """
        # Parse region
        chrom, start, end, strand = parsed_index

        chrom_size = self.sequence_loader.chromsizes[chrom] if self.sequence_loader.chromsizes is not None else inf

        # Get default stochastic shift
        shift = super()._get_shift(**kwargs)

        # Account for negative strand also flipping shift direction
        real_shift = -shift if strand == "-" else shift

        # Check if within genomic boundaries, reroll otherwise
        while (start + real_shift) <= 0 or (end + real_shift) >= chrom_size:
            shift = super()._get_shift(**kwargs)
            real_shift = -shift if strand == "-" else shift

        # Undo negative strand flipping
        shift = -real_shift if strand == "-" else real_shift
        return shift

    def _parse_index(self, index: str) -> tuple[str, int, int, str]:
        """Parse an index into an easier-to-use format.

        For BaseGenomicDataWrapper, we parse to (chrom, start, end, strand) tuples with `parse_region`.
        This is useful since we have to parse to check the genomic boundaries during _get_shift anyway,
        and it can be useful for inheriting classes in _get_sequence/_get_target/etc.
        """
        # Run any parse_region of base class (which likely doesn't exist)
        index = super()._parse_index(index)
        return parse_region(index)

    def _unparse_index(self, index: tuple[str, int, int, str]) -> str:
        """Turn a parsed index back into an indexable state (str, int, etc) from the parsed state."""
        chrom, start, end, strand = index
        return f"{chrom}:{start}-{end}:{strand}"

    def get_config(self) -> dict:
        """Return a dict of properties, to be logged during training.

        Primarily used in Crested.fit().
        """
        config = super().get_config()
        config.update({
            "in_memory": self.sequence_loader.in_memory,
        })
        return config


def recursive_tensor_spec(output):
    """Generate (a tuple) of TensorSpecs recursively, to turn a tuple of (tuple of) arrays into TensorSpecs for tf.data.dataset.from_generator().

    Works on standard (seq, target) dataloader tuples, but also on more complicated things like (seq, (target1, target2)).
    """
    if hasattr(output, 'dtype') and hasattr(output, 'shape'): # proxy for tensor/numpy array/numpy scalar, which are the likely final values
        return tf.TensorSpec(shape=output.shape, dtype=output.dtype)
    else: # Otherwise (if list, tuple or generator), call recursively
        return tuple(recursive_tensor_spec(xi) for xi in output)

def recursive_shape(output):
    """Generate (a tuple) of array shapes recursively.

    Works on standard (seq, target) dataloader tuples, but also on more complicated things like (seq, (target1, target2)).
    """
    if hasattr(output, 'shape'):
        return output.shape
    else:
        return tuple(recursive_shape(xi) for xi in output)

def recursive_move_device(output: keras.KerasTensor | tuple[keras.KerasTensor, ...], device, **kwargs):
    """Move (a tuple) of tensors shapes to another device recursively.

    Works on standard (seq, target) dataloader tuples, but also on more complicated things like (seq, (target1, target2)).
    """
    if isinstance(output, tuple):
        return tuple(recursive_move_device(xi, device) for xi in output)
    elif isinstance(output, list):
        return [recursive_move_device(xi, device) for xi in output]
    else:
        return output.to(device, **kwargs)
