import numpy as np
from anndata import AnnData
from scipy.sparse import spmatrix

from crested._genome import Genome
from crested.tl.data import TrackData

from ._datawrapper import BaseGenomicDataWrapper


class TrackDataWrapper(BaseGenomicDataWrapper):
    def __init__(
        self,
        data: TrackData,
        regions: list[str],
        splits: list[str],
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
        split_column = 'split',
        **kwargs
    ):
        """""" # TODO: ADD DOCS
        # Set some basic values (esp those required for _get_indices and _get_splits)
        self.data = data
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
            train_values=train_values,
            val_values=val_values,
            test_values=test_values,
            **kwargs
        )

    def _get_indices(self):
        """"""
        return self.regions

    def _get_splits(self, split):
        """"""
        return self.splits

    def _get_target(self, expanded_index: str, **kwargs) -> np.ndarray:
        """Get target for a given index."""
        return self.data[expanded_index]

    def __repr__(self):
        return f"TrackDataWrapper: (n_samples={len(self)}, batch_size={self.batch_size}, batched_length={self.batched_length()}, data shape: {self.data.shape}, input_shape={self.input_shape}, output_shape={self.output_shape})" #TODO: finish


class GeckoDataWrapper(BaseGenomicDataWrapper):
    def __init__(
        self,
        anndata: AnnData,
        trackdata: TrackData,
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
        split_column = 'split', # TODO: add docs for these, maybe only keep this in downstream implementations
        **kwargs
    ): # TODO: ADD ARGS
        """"""
        # Set some basic values (esp those required for _get_indices and _get_splits)
        self.data = anndata
        self.trackdata = trackdata
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
            train_values=train_values,
            val_values=val_values,
            test_values=test_values,
            **kwargs
        )

        # Set some last variables dependent on having indices or extracting sequences
        self.index_map = {index: i for i, index in enumerate(self.indices)}

    def _get_indices(self):
        """"""
        return list(self.data.var_names)

    def _get_splits(self, split):
        """"""
        return list(self.data.var[self.split_column])

    def _get_target(self, original_index: str, expanded_index: str, **kwargs) -> np.ndarray:
        """Get target for a given index."""
        y_index = self.index_map[original_index]
        y1 = (
            self.data.X[:, y_index].toarray().flatten()
            if self.compressed
            else self.data.X[:, y_index].astype('float32')
        )
        y2 = self.trackdata[expanded_index]
        return y1, y2

    def __repr__(self):
        return f"TrackAnnDataWrapper: (n_samples={len(self)}, batch_size={self.batch_size}, batched_length={self.batched_length()}, data shape: {self.data.shape}, input_shape={self.input_shape}, output_shape={self.output_shape})" #TODO: finish
