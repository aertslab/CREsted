"""Dataloader for batching, shuffling, and one-hot encoding of AnnDataset."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from anndata import AnnData

from ._dataset import AnnDataset

BASE_TO_INT_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}
STATIC_HASH_TABLE = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(BASE_TO_INT_MAPPING.keys())),
        values=tf.constant(list(BASE_TO_INT_MAPPING.values()), dtype=tf.int32),
    ),
    default_value=-1,
)


class AnnDataLoader:
    """
    DataLoader class for AnnDataset with options for batching, shuffling, and one-hot encoding.

    Attributes
    ----------
    dataset
        The dataset instance provided.
    batch_size
        Number of samples per batch.
    shuffle
        Indicates whether shuffling is enabled.
    one_hot_encode
        Indicates whether one-hot encoding is enabled.
    drop_remainder
        Indicates whether to drop the last incomplete batch.

    Examples
    --------
    >>> dataset = AnnDataset(...)  # Your dataset instance
    >>> batch_size = 32
    >>> dataloader = AnnDataLoader(
    ...     dataset, batch_size, shuffle=True, one_hot_encode=True, drop_remainder=True
    ... )
    >>> for x, y in dataloader.data:
    ...     # Your training loop here
    """

    def __init__(
        self,
        dataset: AnnDataset,
        batch_size: int,
        shuffle: bool = False,
        one_hot_encode: bool = True,
        drop_remainder: bool = True,
    ):
        """
        Initialize the DataLoader with the provided dataset and options.

        Parameters
        ----------
        dataset
            An instance of AnnDataset containing the data to be loaded.
        batch_size
            Number of samples per batch to load.
        shuffle
            If True, the data will be shuffled at the end of each epoch. Default is False.
        one_hot_encode
            If True, sequences will be one-hot encoded. Default is True.
        drop_remainder
            If True, the last batch will be dropped if it is smaller than batch_size. Default is True.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.one_hot_encode = one_hot_encode
        self.drop_remainder = drop_remainder

        if self.shuffle:
            self.dataset.shuffle = True

    @tf.function
    def _map_one_hot_encode(self, sequence, target):
        """One hot encoding as a tf mapping function during prefetching."""
        if isinstance(sequence, str):
            sequence = tf.constant([sequence])
        elif isinstance(sequence, tf.Tensor) and sequence.ndim == 0:
            sequence = tf.expand_dims(sequence, 0)

        def one_hot_encode(sequence):
            char_seq = tf.strings.unicode_split(sequence, "UTF-8")
            integer_seq = STATIC_HASH_TABLE.lookup(char_seq)
            x = tf.one_hot(integer_seq, depth=4)
            return x

        one_hot_sequence = tf.map_fn(
            one_hot_encode,
            sequence,
            fn_output_signature=tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        )
        one_hot_sequence = tf.squeeze(one_hot_sequence, axis=0)  # remove extra map dim
        return one_hot_sequence, target

    def _create_dataset(self):
        ds = tf.data.Dataset.from_generator(
            lambda: self.dataset,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(self.dataset.num_outputs,), dtype=tf.float32),
            ),
        )
        if self.one_hot_encode:
            ds = ds.map(
                lambda seq, tgt: self._map_one_hot_encode(seq, tgt),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        ds = (
            ds.batch(self.batch_size, drop_remainder=self.drop_remainder)
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )
        return ds

    @property
    def data(self):
        return self._create_dataset()

    def __len__(self):
        return len(self.dataset) // self.batch_size


if __name__ == "__main__":
    # Test the dataloader
    # TODO: remove
    import pandas as pd
    import scipy.sparse as sp

    from crested import import_topics
    from crested.pp import train_val_test_split
    from crested.tl.data import AnnDataset

    def create_anndata_with_regions(
        regions: list[str],
        chr_var_key: str = "chr",
        compress: bool = False,
        random_state: int = None,
    ) -> AnnData:
        if random_state is not None:
            np.random.seed(random_state)
        data = np.random.randn(3, len(regions))
        var = pd.DataFrame(index=regions)
        var[chr_var_key] = [region.split(":")[0] for region in regions]
        var["start"] = [int(region.split(":")[1].split("-")[0]) for region in regions]
        var["end"] = [int(region.split(":")[1].split("-")[1]) for region in regions]

        if compress:
            data = sp.csr_matrix(data)

        return AnnData(X=data, var=var)

    genome_file = "/staging/leuven/res_00001/genomes/10xgenomics/CellRangerARC/refdata-cellranger-arc-mm10-2020-A-2.0.0/fasta/genome.fa"

    regions = [
        "chr1:3094805-3095305",
        "chr1:3095470-3095970",
        "chr1:3112174-3112674",
        "chr1:3113534-3114034",
        "chr1:3119746-3120246",
        "chr1:3120272-3120772",
        "chr1:3121251-3121751",
        "chr1:3134586-3135086",
        "chr1:3165708-3166208",
        "chr1:3166923-3167423",
    ]
    adata = import_topics(
        topics_folder="/staging/leuven/stg_00002/lcb/lmahieu/projects/DeepTopic/biccn_test/otsu",
        regions_file="/staging/leuven/stg_00002/lcb/lmahieu/projects/DeepTopic/biccn_test/consensus_peaks_bicnn.bed",
        compress=False,
        # topics_subset=["topic_1", "topic_2"], # optional subset of topics to import
    )
    genome_file = "/staging/leuven/res_00001/genomes/10xgenomics/CellRangerARC/refdata-cellranger-arc-mm10-2020-A-2.0.0/fasta/genome.fa"

    # adata = create_anndata_with_regions(regions, compress=False, random_state=42)
    train_val_test_split(
        adata,
        strategy="region",
        val_size=0.1,
        test_size=0.1,
        shuffle=True,
        random_state=42,
    )
    import time

    # Test anndataset
    train_data = AnnDataset(
        adata,
        genome_file,
        split="train",
        always_reverse_complement=True,
    )
    print(train_data)
    train_loader = AnnDataLoader(train_data, shuffle=True, batch_size=256)
    print(f"LENGTH DATA: {len(train_data)}")
    print(f"LENGTH LOADER: {len(train_loader)}")
    # start = time.time()
    # for i, (x, y) in enumerate(dataset):
    #     if i == 100000:
    #         break
    #     if i % 10000 == 0:
    #         print("Time taken:", time.time() - start)
    #         start = time.time()

    # # # # test dataloader

    # # time the code

    start = time.time()
    for i, (x, y) in enumerate(train_loader.data):
        # if i % 50 == 0:
        #     print("Time taken:", time.time() - start)
        #     start = time.time()
        # print(i)
        if i == 300:
            print(x.shape, y.shape)
            print("done")
            break
