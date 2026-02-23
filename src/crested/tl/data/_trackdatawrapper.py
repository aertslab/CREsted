import numpy as np
from anndata import AnnData

from crested._genome import Genome
from crested.tl.data import TrackData

from ._anndatawrapper import AnnDataWrapper
from ._datawrapper import BaseGenomicDataWrapper


class TrackDataWrapper(BaseGenomicDataWrapper):
    def __init__(
        self,
        data: TrackData,
        regions: list[str],
        splits: list[str],
        genome: Genome | None = None,
        batch_size: int = 32,
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
        """""" # TODO: ADD DOCS
        # Set some basic values (esp those required for _get_indices and _get_splits)
        self.trackdata = data
        self.regions = regions
        self.splits = splits

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

    def _get_indices(self):
        """"""
        return self.regions

    def _get_splits(self):
        """"""
        return self.split

    def _get_shift(self, **kwargs) -> int:
        """Return a stochastic shift value.

        If the trackdata is pre-binned, rounds to the nearest fitting bin size.
        """
        shift = super()._get_shift(**kwargs)
        if self.trackdata.prebinned:
            shift = int(np.round(shift / self.trackdata.bin_size) * self.trackdata.bin_size)
        return shift

    def _get_target(self, parsed_index: tuple[str, int, int, str], revcomp: bool, shift: int, **kwargs) -> np.ndarray:
        """Get target for a given index."""
        track = self.data.get_track(parsed_index, shift=shift)
        if revcomp:
            track = np.flip(track, axis=1)
        return track

    def _get_expanded_index(self, index):
        """Get an expanded index from on the original index.

        Apply inherited transformations from BaseGenomicDataWrapper.
        If the trackdata object is prebinned, rounds the start and end to the nearest binsize.
        """
        # Apply transform of inheriting object (generally adding a strand)
        expanded_index = super()._get_expanded_index(index)

        # If pre-binned: align with bin edges
        if self.trackdata.prebinned:
            # Parse transformed into values
            chrom, start, end, strand = self._parse_index(expanded_index)
            # Shift values
            seq_len = end-start
            start = int(np.round(start / self.trackdata.bin_size) * self.trackdata.bin_size)
            end = start+seq_len
            # Transform back into string
            expanded_index = self._unparse_index((chrom, start, end, strand))
        return expanded_index

    def __repr__(self):
        return f"TrackDataWrapper: (n_samples={len(self)}, batch_size={self.batch_size}, batched_length={self.batched_length()}, data shape: {self.trackdata.shape}, input_shape={self.input_shape}, output_shape={self.output_shape})" #TODO: finish

# Inherit AnnDataWrapper for anndata support w.r.t. indexing, sequence retrieval, etc. and add TrackDataWrapper things manually
# I tried dual inheritance but it makes the AnnDataWrapper's super().__init__ call TrackDataWrapper's, which is not what we want
class GeckoDataWrapper(AnnDataWrapper):
    def __init__(
        self,
        anndata: AnnData,
        trackdata: TrackData,
        genome: Genome | None = None,
        batch_size: int = 32,
        random_reverse_complement: bool = False,
        always_reverse_complement: bool = True,
        max_stochastic_shift: int = 0,
        in_memory: bool = True,
        drop_remainder: bool = False,
        train_splits: str | list = 'train',
        val_splits: str | list = 'val',
        test_splits: str | list = 'test',
        split_column = 'split', # TODO: add docs for these, maybe only keep this in downstream implementations
        **kwargs
    ): # TODO: ADD ARGS
        """"""
        # Set track-specific things not set by AnnDataWrapper's init
        self.trackdata = trackdata

        # Initialize AnnDataWrapper functionality (handle anndata indexing and through to genomicdatawrapper functionality)
        super().__init__(
            data=anndata,
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
            split_column=split_column,
            **kwargs
        )

        # Set some last variables dependent on having indices or extracting sequences
        self.index_map = {index: i for i, index in enumerate(self.indices)}

    def _get_indices(self):
        """"""
        return list(self.data.var_names)

    def _get_splits(self):
        """"""
        return list(self.data.var[self.split_column])

    def _get_target(self, original_index: str, parsed_index: str, revcomp: bool, shift: int, **kwargs) -> np.ndarray:
        """Get target for a given index."""
        # Get track
        y1 = self.trackdata.get_track(parsed_index, shift=shift)
        if revcomp:
            y1 = np.flip(y1, axis=1)

        # Get scalar
        y_index = self.index_map[original_index]
        y2 = (
            self.data.X[:, y_index].toarray().flatten()
            if self.compressed
            else self.data.X[:, y_index].astype('float32')
        )
        return y1, y2

    def _get_shift(self, **kwargs) -> int:
        """Return a stochastic shift value.

        If the trackdata is pre-binned, rounds to the nearest fitting bin size.
        """
        shift = super()._get_shift(**kwargs)
        if self.trackdata.prebinned:
            shift = int(np.fix(shift / self.trackdata.bin_size) * self.trackdata.bin_size)
        return shift

    def _get_expanded_index(self, index):
        """Get an expanded index from on the original index.

        Apply inherited transformations from BaseGenomicDataWrapper.
        If the trackdata object is prebinned, rounds the start and end to the nearest binsize.
        """
        # Apply transform of inheriting object (generally adding a strand)
        expanded_index = super()._get_expanded_index(index)

        # If pre-binned: align with bin edges
        if self.trackdata.prebinned:
            # Parse transformed into values
            chrom, start, end, strand = self._parse_index(expanded_index)
            # Shift values
            seq_len = end-start
            start = int(np.round(start / self.trackdata.bin_size) * self.trackdata.bin_size)
            end = start+seq_len
            # Transform back into string
            expanded_index = self._unparse_index((chrom, start, end, strand))
        return expanded_index

    def __repr__(self):
        return f"GeckoDataWrapper: (n_samples={len(self)}, batch_size={self.batch_size}, batched_length={self.batched_length()}, input_shape={self.input_shape}, output_shape={self.output_shape})" #TODO: finish
