""""""

from __future__ import annotations

import os
import re
from math import ceil

import keras
import numpy as np
from loguru import logger
from tqdm import tqdm

if os.environ["KERAS_BACKEND"] == "torch":
    import torch
    from torch.utils.data import DataLoader
    FrameworkDatasetClass = torch.utils.data.Dataset
else:
    import tensorflow as tf
    FrameworkDatasetClass = object

from crested._genome import Genome, _resolve_genome
from crested.utils import one_hot_encode_sequence


class BaseDataWrapper:
    """
    Base DataWrapper class for retrieving sequences and their associated target values.

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
        If True, all sequences will be augmented with their reverse complement during training.
        Incompatible with random_reverse_complement.
    max_stochastic_shift
        Maximum stochastic shift (n base pairs in either direction) to apply randomly to each sequence during training.
        Default is 0 (disabled).
    drop_remainder
        If True, drop the last batch if it is not the full batch_size. Default is False.
    train_values
        The values in your split labeling that correspond to the training set as string or list of strings, i.e 'train' or ['fold0', 'fold1', 'fold2']
    val_values
        The values in your split labeling that correspond to the validation set as string or list of strings, i.e 'val' or ['fold3', 'fold4']
    test_values
        The values in your split labeling that correspond to the test set as string or list of strings, i.e 'test' or ['fold5', 'fold6']
    """

    # ----- Object initialization -----
    def __init__(
        self,
        batch_size: int = 256,
        random_reverse_complement: bool = False,
        always_reverse_complement: bool = True,
        max_stochastic_shift: int = 0,
        drop_remainder: bool = False,
        train_values: str | list = 'train',
        val_values: str | list = 'val',
        test_values: str | list = 'test',
    ):
        """Initialize the DataWrapper with the provided dataset and options."""
        # Dataset
        # Split parameters
        self.split_values = {
            'train': [train_values] if isinstance(train_values, str) else train_values,
            'val': [val_values] if isinstance(val_values, str) else val_values,
            'test': [test_values] if isinstance(test_values, str) else test_values,
        }
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
        if os.environ["KERAS_BACKEND"] == "torch":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Process indices
        self.indices = self._get_indices()
        self.split_indices = {split: self._split_indices(split) for split in  ('train', 'val', 'test')}
        # Check if indices (look like they) contain stranded information
        # Note: don't use _check_region_strandedness since that also checks for region formatting, while we want to allow all kinds of indices here
        if isinstance(self.indices[0], str) and self.indices[0][-1] in ('+', '-'):
            self.stranded = True
        else:
            self.stranded = False

        # Set up sequence index handling to match deterministic augmentation (always_reverse_complement) to original output values
        # Pass all values through _expand_indices (since it makes sure all values are stranded), for training also actually augment (adds always_rev_comp)

        # Get stranded indices for entire dataset (for prediction)
        self.split_expanded_indices = {}
        self.split_expanded_indices['predict'] = self._expand_indices(self.indices, augment_revcomp = False)
        # Get stranded indices per split, augmenting for training set
        self.split_expanded_indices['train'] =  self._expand_indices(self.split_indices['train'], augment_revcomp = self.always_reverse_complement)
        for split in ('val', 'test'):
            self.split_expanded_indices[split] =  self._expand_indices(self.split_indices[split], augment_revcomp = False)
        # Get total set of augmented indices across all possibilities
        self.full_expanded_indices = {aug_index for aug_index_list in self.split_expanded_indices.values() for aug_index in aug_index_list}

    # ----- Primary methods to implement/overwrite in the inherited versions: -----
    def _get_indices(self):
        """Get a list of all indices. Indices (possibly after augmentation) are used in _get_sequence() and _get_target() to retrieve sequence and targets for that entry, respectively."""
        # Pseudocode: return self.data.metadata.index
        raise NotImplementedError("Please define `self._get_indices()` to extract some kind of unique ID per sample from your dataset, i.e. AnnData var_names or DataFrame index.")

    def _get_splits(self):
        """Get values that map to train/val/test splits for each index.

        Expected to return values according to train/val/test_values, so if you have train_values='fold0', val_values='fold1', test_values='fold2',
        _get_splits should return values of {'fold0', 'fold1', 'fold2'}.
        """
        # Pseudocode: return self.data.metadata[self.split_column]
        raise NotImplementedError("Please define `self._get_splits()` to extract a split identifier per sample from your dataset, i.e. AnnData var['split'].")

    def _get_sequence(self, original_index: str, expanded_index: str, revcomp: bool = False, shift: int = 0, **kwargs) -> str:
        """Get a sequence (as a string) given a certain index."""
        raise NotImplementedError("Implement _get_sequence of your inherited DataWrapper class to retrieve (unencoded) input sequences for your classes")

    def _encode_sequence(self, seq: str, **kwargs) -> np.ndarray:
        """Encode the sequence as string to a numerical representation by one-hot encoding it. Returned value should not have a batch dimension yet."""
        return one_hot_encode_sequence(seq, expand_dim=False)

    def _get_target(self, original_index: str, expanded_index: str, revcomp: bool = False, shift: int = 0, **kwargs) -> np.ndarray:
        """Get target for a given index. Returned value should not have a batch dimension yet."""
        raise NotImplementedError("Implement _get_target() of your inherited DataWrapper class to retrieve output values for your sequence")


    # ----- Index management -----
    def _split_indices(self, split: str) -> list[str]:
        """Split the list of indices according to the split they belong to."""
        return [index for index, index_split in zip(self.indices, self._get_splits()) if index_split in self.split_values[split]]

    def _expand_indices(self, indices: list[str], augment_revcomp: bool) -> list[str]:
        """Add strand information to indices, if not already present. Optionally also augments total set of indices by adding the reverse complement version index."""
        if not hasattr(self, 'expanded_indices_map'):
            self.expanded_indices_map = {}
        expanded_indices = []
        for region in indices:
            if not self.stranded:
                stranded_region = f"{region}:+"
            else:
                stranded_region = region
            expanded_indices.append(stranded_region)
            self.expanded_indices_map[stranded_region] = region # Update shared backmapping with the original indices
            if augment_revcomp:
                expanded_indices.append(_flip_region_strand(stranded_region))
                self.expanded_indices_map[_flip_region_strand(stranded_region)] = region
        return expanded_indices


    # ----- Item retrieval -----
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return sequence and target for a given numeric index. Not used too much: primary indexing function is get_indexed_item().

        Main logic is implemented in get_indexed_item() to retrieve using expanded indices and with optional augmentation.
        """
        # Get expanded index (after adding revcomp indices)
        # Generally a guaranteed-stranded version of the original index.
        expanded_index = self.split_expanded_indices['predict'][idx]
        return self.get_indexed_item(expanded_index)

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

        # Get stochastic shift amount
        if augment and self.max_stochastic_shift > 0:
            shift = np.random.randint(
                -self.max_stochastic_shift, self.max_stochastic_shift + 1
            )
        else:
            shift = 0
        # Get whether to random reverse complement (always_reverse_complement is done in the sequence loader by encoding the strand in the index)
        if augment and self.random_reverse_complement and np.random.rand() < 0.5:
            revcomp = True
        else:
            revcomp = False

        # Get sequence as string and apply stochastic shift/revcomp augmentations, letting the function choose
        # whether to use original (like when grabbing from dataframe) or expanded index (like when extracting from genome)
        x = self._get_sequence(original_index=original_index, expanded_index=expanded_index, revcomp=revcomp, shift=shift)

        # Encode sequence (one-hot by default, but can override for i.e. tokenisation)
        x = self._encode_sequence(x)

        # Get targets, letting the function choose whether to use original (like with scalars) or augmented index (like with tracks)
        y = self._get_target(original_index=original_index, expanded_index=expanded_index, revcomp=revcomp, shift=shift)

        return x, y

    # ----- Item batching -----
    def _collate_fn(self, batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[keras.KerasTensor], list[keras.KerasTensor]]:
        """Collate function to combine entries into a batch and move tensors to the specified device if backend is torch.

        Should work as is if your input and output are both a single tensor. If any are tuples or dicts of tensors, adjust inputs/targets respectively.
        """
        # TODO: make this work on nested tensors out of the box as well, so that people don't have to adjust this?
        inputs, targets = zip(*batch)
        inputs = torch.stack([torch.tensor(input) for input in inputs]).to(self.device)
        targets = torch.stack([torch.tensor(target) for target in targets]).to(self.device)
        return inputs, targets

    # ----- Dataloader creation -----
    def create_dataloader(self, split: str | None, augment: bool = False, shuffle: bool = False, **kwargs):
        """"""

        # PyTorch loops based on __len__ and __getitem__
        if os.environ["KERAS_BACKEND"] == "torch":
            return DataLoader(
                self._create_looper(split=split, augment=augment),
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=self.drop_remainder,
                num_workers=0, # TODO: investigate whether we should adjust this to some autotune amount
                collate_fn=self._collate_fn,
            )
        # TensorFlow loops using a generator
        elif os.environ["KERAS_BACKEND"] == "tensorflow":
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

    def _create_looper(self, split: str, augment: bool = False, shuffle_generator: bool = False) -> DataLooper:
        return DataLooper(datawrapper=self, indices=self.split_expanded_indices[split], augment=augment, shuffle_generator=shuffle_generator)

    @property
    def train_dataloader(self):
        """Return a tf.data.Dataset or torch.utils.data.DataLoader instance of the training samples, augmented."""
        return self.create_dataloader('train', augment=True, shuffle=True)
    @property
    def val_dataloader(self):
        """Return a tf.data.Dataset or torch.utils.data.DataLoader instance of the validation samples.."""
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

    def split_len(self, split) -> int:
        """"""
        return len(self.split_expanded_indices[split])

    @property
    def input_shape(self):
        if self.input_shape_cache is None:
            self.input_shape_cache = recursive_shape(self[0][0])
        return self.input_shape_cache

    @property
    def output_shape(self):
        if self.output_shape_cache is None:
            self.output_shape_cache = recursive_shape(self[0][1])
        return self.output_shape_cache

    def __len__(self) -> int:
        """Get number of (augmented) samples in the datawrapper."""
        return len(self.split_expanded_indices['predict'])

    def __repr__(self) -> str:
        """Get string representation of the wrapped dataset."""
        return f"{self.__class__.__name__}: (n_samples={len(self)},  batch_size={self.batch_size}, batched_length={self.batched_length()}" # TODO: expand (without self.data? or just leave to inheriting classes to figure out)


class DataLooper(FrameworkDatasetClass):
    """A mini-class providing iteration and indexing based on the subset of indices provided."""

    def __init__(
        self,
        datawrapper,
        indices: list[str],
        augment: bool,
        shuffle_generator: bool = False
    ):
        """"""
        self.datawrapper = datawrapper
        self.indices = indices
        self.augment = augment
        self.shuffle_generator = shuffle_generator


    # ----- PyTorch indexing side -----
    def __getitem__(self, idx: int):
        return self.datawrapper.get_indexed_item(self.indices[idx], augment=self.augment)

    def __len__(self):
        return len(self.indices)

    def _collate_fn(self, batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[keras.KerasTensor], list[keras.KerasTensor]]:
        """Collate function for PyTorch. Uses the base datawrapper's implementation."""
        return self.datawrapper._collate_fn(batch)

    # ----- TensorFlow generator side -----
    def get_generator(self):
        loop_indices = self.indices
        if self.shuffle_generator:
            rng = np.random.default_rng()
            loop_indices = rng.permutation(loop_indices, axis = 0)
        for index in loop_indices:
            yield self.datawrapper.get_indexed_item(index, augment=self.augment)

class BaseGenomicDataWrapper(BaseDataWrapper):
    """A slightly expanded BaseDataWrapper, with genome management built-in."""

    def __init__(
        self,
        genome: Genome | None = None,
        batch_size: int = 256,
        random_reverse_complement: bool = False,
        always_reverse_complement: bool = True,
        max_stochastic_shift: int = 0,
        in_memory: bool = True,
        drop_remainder: bool = False,
        train_values: str | list = 'train',
        val_values: str | list = 'val',
        test_values: str | list = 'test',
        **kwargs
    ):
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
            If True, all sequences will be augmented with their reverse complement during training.
            Incompatible with random_reverse_complement.
        max_stochastic_shift
            Maximum stochastic shift (n base pairs in either direction) to apply randomly to each sequence during training.
            Default is 0 (disabled).
        in_memory
            If True, extract the sequences from the genome before training starts and save them to memory,
            rather than extracting on the fly every time. Default is True.
        drop_remainder
            If True, drop the last batch if it is not the full batch_size. Default is False.
        train_values
            The values in your split labeling that correspond to the training set as string or list of strings, i.e 'train' or ['fold0', 'fold1', 'fold2']
        val_values
            The values in your split labeling that correspond to the validation set as string or list of strings, i.e 'val' or ['fold3', 'fold4']
        test_values
            The values in your split labeling that correspond to the test set as string or list of strings, i.e 'test' or ['fold5', 'fold6']
        kwargs
            Remaining keyword arguments, passed to BaseDataLoader.
        """
        super().__init__(
            batch_size=batch_size,
            random_reverse_complement=random_reverse_complement,
            always_reverse_complement=always_reverse_complement,
            max_stochastic_shift=max_stochastic_shift,
            drop_remainder=drop_remainder,
            train_values=train_values,
            val_values=val_values,
            test_values=test_values,
            **kwargs
        )
        genome =  _resolve_genome(genome)

        # Check region formatting: do they match chr:start-end[:strand] format and if so, do they contain strand info
        _ = (_check_region_strandedness(index) for index in self.indices)

        self.sequence_loader = SequenceLoader(
            genome,
            in_memory=in_memory,
            always_reverse_complement=self.always_reverse_complement,
            max_stochastic_shift=self.max_stochastic_shift,
            regions=self.indices,
        )

        # Save input shape now that we've created the function to retrieve one
        # Also pass original_index just in case we override _get_sequence in an inheriting class and use it
        example_index = self.split_expanded_indices['predict'][0]
        self.input_shape_cache = self._encode_sequence(self._get_sequence(
            expanded_index = example_index,
            original_index = self.expanded_indices_map[example_index],
        )).shape

        if self.input_shape_cache[-1] > self.input_shape_cache[-2]:
            logger.warning(
                "Keras (and therefore CREsted) expects sequences to be encoded as (seq_len, n_nucleotides)."
                f"Your input shape appears to be flipped: {self.input_shape_cache}"
            )

    def _get_sequence(self, expanded_index: str, revcomp: bool = False, shift: int = 0, **kwargs) -> np.ndarray:
        """Get a sequence (as a string) given a certain index."""
        # We need the stranded information when extracting from the genome so use the expanded index
        x = self.sequence_loader.get_sequence(
            expanded_index, stranded=True, shift=shift
        )
        if revcomp:
            x = self.sequence_loader._reverse_complement(x)
        return x


def _flip_region_strand(region: str) -> str:
    """Reverse the strand of a region."""
    strand_reverser = {"+": "-", "-": "+"}
    return region[:-1] + strand_reverser[region[-1]]


def _check_region_strandedness(region: str) -> bool:
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

class SequenceLoader:
    """
    Load sequences from a genome file.

    Options for reverse complementing and stochastic shifting are available.

    Parameters
    ----------
    genome
        Genome instance.
    in_memory
        If True, the sequences of supplied regions will be loaded into memory.
    always_reverse_complement
        If True, all sequences will be augmented with their reverse complement.
        Doubles the dataset size.
    max_stochastic_shift
        Maximum stochastic shift (n base pairs) to apply randomly to each sequence.
    regions
        List of regions to load into memory. Required if in_memory is True.
    """

    def __init__(
        self,
        genome: Genome,
        in_memory: bool = False,
        always_reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
        regions: list[str] | None = None,
    ):
        """Initialize the SequenceLoader with the provided genome file and options."""
        self.genome = genome.fasta
        self.chromsizes = genome.chrom_sizes
        self.in_memory = in_memory
        self.always_reverse_complement = always_reverse_complement
        self.max_stochastic_shift = max_stochastic_shift
        self.sequences = {}
        self.complement = str.maketrans("ACGT", "TGCA")
        self.regions = regions

        if self.in_memory:
            self._load_sequences_into_memory(self.regions)
        # TODO: maybe add check for sequence length 

    def _load_sequences_into_memory(self, regions: list[str]):
        """Load all sequences into memory (dict)."""
        logger.info("Loading sequences into memory...")
        # Check region formatting
        stranded = _check_region_strandedness(regions[0])

        for region in tqdm(regions):
            # Make region stranded if not
            if not stranded:
                strand = "+"
                region = f"{region}:{strand}"
                if region[-4] == ":":
                    raise ValueError(
                        f"You are double-adding strand ids to your region {region}. Check if all regions are stranded or unstranded."
                    )

            # Parse region
            chrom, start_end, strand = region.split(":")
            start, end = map(int, start_end.split("-"))

            # Add region to self.sequences
            extended_sequence = self._get_extended_sequence(
                chrom, start, end, strand
            )
            self.sequences[region] = extended_sequence

            # Add reverse-complemented region to self.sequences if always_reverse_complement
            if self.always_reverse_complement:
                self.sequences[
                    _flip_region_strand(region)
                ] = self._reverse_complement(extended_sequence)
            # TODO: maybe pass augmented regions to function instead and then we don't have to do this again here, also don't have to add strand to regions above
            # TODO: change that and also change it in the original dataset code, small change that shouldn't break things as long as it's synchronised
            # TODO: problem: don't have combined set of augmented train and non-augmented val/test set.

    def _get_extended_sequence(
        self, chrom: str, start: int, end: int, strand: str
    ) -> str:
        """Get sequence from genome file, extended for stochastic shifting."""
        extended_start = max(0, start - self.max_stochastic_shift)
        extended_end = extended_start + (end - start) + (self.max_stochastic_shift * 2)

        if self.chromsizes and chrom in self.chromsizes:
            chrom_size = self.chromsizes[chrom]
            if extended_end > chrom_size:
                extended_start = chrom_size - (
                    end - start + self.max_stochastic_shift * 2
                )
                extended_end = chrom_size

        seq = self.genome.fetch(chrom, extended_start, extended_end).upper()
        if strand == "-":
            seq = self._reverse_complement(seq)
        return seq

    def _reverse_complement(self, sequence: str) -> str:
        """Reverse complement a sequence."""
        return sequence.translate(self.complement)[::-1]

    def get_sequence(
        self, region: str, stranded: bool | None = None, shift: int = 0
    ) -> str:
        """
        Get sequence for a region, strand, and shift from memory or fasta.

        If no strand is given in region or strand, assumes positive strand.

        Parameters
        ----------
        region
            Region to get the sequence for. Either (chr:start-end) or (chr:start-end:strand).
        stranded
            Whether the input data is stranded. Default (None) infers from sequence (at a computational cost).
            If not stranded, positive strand is assumed.
        shift:
            Shift of the sequence within the extended sequence, for use with the stochastic shift mechanism.

        Returns
        -------
        The DNA sequence, as a string.
        """
        if stranded is None:
            stranded = _check_region_strandedness(region)
        if not stranded:
            region = f"{region}:+"
        # Parse region
        chrom, start_end, strand = region.split(":")
        start, end = map(int, start_end.split("-"))

        # Get extended sequence
        if self.in_memory:
            sequence = self.sequences[region]
        else:
            sequence = self._get_extended_sequence(chrom, start, end, strand)

        # Extract from extended sequence
        start_idx = self.max_stochastic_shift + shift
        end_idx = start_idx + (end - start)
        sub_sequence = sequence[start_idx:end_idx]

        # Pad with Ns if sequence is shorter than expected
        if len(sub_sequence) < (end - start):
            if strand == "+":
                sub_sequence = sub_sequence.ljust(end - start, "N")
            else:
                sub_sequence = sub_sequence.rjust(end - start, "N")

        return sub_sequence

def recursive_tensor_spec(output):
    """Generate (a tuple) of TensorSpecs recursively, to turn a tuple of (tuple of) arrays into TensorSpecs for tf.data.dataset.from_generator().

    Works on standard (seq, target) dataloader output tuples, but also on more complicated things like (seq, (target1, target2)).
    """
    if tf.is_tensor(output) or isinstance(output, np.ndarray):
        return tf.TensorSpec(shape=output.shape, dtype=output.dtype)
    else:
        return tuple(recursive_tensor_spec(xi) for xi in output)

def recursive_shape(output):
    """Generate (a tuple) of array shapes recursively.

    Works on standard (seq, target) dataloader output tuples, but also on more complicated things like (seq, (target1, target2)).
    """
    if hasattr(output, 'shape'):
        return output.shape
    else:
        return tuple(recursive_shape(xi) for xi in output)
