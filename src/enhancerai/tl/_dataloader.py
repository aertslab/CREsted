from __future__ import annotations

from os import PathLike

import numpy as np
import tensorflow as tf
from anndata import AnnData
from pysam import FastaFile
from scipy.sparse import spmatrix
from tqdm import tqdm


class AnnDataSet:
    def __init__(
        self,
        anndata: AnnData,
        genome_file: PathLike,
        indices: list[str],
        chromsizes_file: PathLike | None = None,
        in_memory: bool = True,
        random_reverse_complement: bool = False,
        always_reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
    ):
        self.anndata = anndata
        self.indices = list(indices)
        self.in_memory = in_memory
        self.compressed = isinstance(self.anndata.X, spmatrix)
        self.genome = FastaFile(genome_file)
        self.chromsizes = chromsizes_file

        self.complement = str.maketrans("ACGT", "TGCA")
        self.random_reverse_complement = random_reverse_complement
        self.always_reverse_complement = always_reverse_complement

        if random_reverse_complement and always_reverse_complement:
            raise ValueError(
                "Only one of `random_reverse_complement` and `always_reverse_complement` can be True."
            )

        if self.in_memory:
            print("Loading sequences into memory...")
            self.sequences = {}
            for region in tqdm(self.indices):
                self.sequences[f"{region}:+"] = self._get_sequence(region)
                if self.always_reverse_complement:
                    self.sequences[f"{region}:-"] = self._reverse_complement(
                        self.sequences[f"{region}:+"]
                    )

        self.augmented_indices, self.augmented_indices_map = self._augment_indices()

    def _augment_indices(self):
        augmented_indices = []
        augmented_indices_map = {}
        for region in self.indices:
            augmented_indices.append(f"{region}:+")
            augmented_indices_map[f"{region}:+"] = region
            if self.always_reverse_complement:
                augmented_indices.append(f"{region}:-")
                augmented_indices_map[f"{region}:-"] = region
        return augmented_indices, augmented_indices_map

    def __len__(self):
        return len(self.augmented_indices)

    def _get_sequence(self, region):
        """Get sequence from genome file"""
        chrom, start_end = region.split(":")
        start, end = start_end.split("-")
        return self.genome.fetch(chrom, int(start), int(end))

    def _reverse_complement(self, sequence):
        return sequence.translate(self.complement)[::-1]

    def _get_target(self, index):
        """Get target values"""
        y_index = self.indices.index(index)
        if self.compressed:
            return self.anndata.X[:, y_index].toarray().flatten()
        return self.anndata.X[:, y_index]

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:
        """Get x, y (seq, target) by index"""
        augmented_index = self.augmented_indices[idx]
        original_index = self.augmented_indices_map[augmented_index]

        if self.in_memory:
            x = self.sequences[augmented_index]
        else:
            x = self._get_sequence(original_index)

            if augmented_index.endswith(":-"):
                # only possible if always_reverse_complement is True
                x = self._reverse_complement(x)

        if self.random_reverse_complement:
            if np.random.rand() < 0.5:
                x = self._reverse_complement(x)

        y = self._get_target(original_index)

        return x, y

    def __iter__(self):
        """Generator for iterating over the dataset"""
        for i in range(len(self)):
            yield self.__getitem__(i)


class AnnDataLoader:
    def __init__(
        self,
        anndata: AnnData,
        genome_file: PathLike,
        chromsizes_file: PathLike | None = None,
    ):
        self.anndata = anndata
        self.genome_file = genome_file
        self.chromsizes_file = chromsizes_file

        # get seq len from region width
        start, end = self.anndata.var_names[0].split(":")[1].split("-")
        self.seq_len = int(end) - int(start)
        print(f"Detected sequence length of {self.seq_len} bp.")

        self.num_outputs = self.anndata.n_obs

        self.base_to_int_mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        self.table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(list(self.base_to_int_mapping.keys())),
                values=tf.constant(
                    list(self.base_to_int_mapping.values()), dtype=tf.int32
                ),
            ),
            default_value=-1,
        )
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str = "fit", **kwargs):
        train_idx = self.anndata.var_names[self.anndata.var["split"] == "train"]
        val_idx = self.anndata.var_names[self.anndata.var["split"] == "val"]
        test_idx = self.anndata.var_names[self.anndata.var["split"] == "test"]
        all_idx = self.anndata.var_names

        if stage == "fit":
            self.train_dataset = AnnDataSet(
                self.anndata,
                self.genome_file,
                train_idx,
                self.chromsizes_file,
                **kwargs,
            )
            self.val_dataset = AnnDataSet(
                self.anndata, self.genome_file, val_idx, self.chromsizes_file
            )
        elif stage == "test":
            self.test_dataset = AnnDataSet(
                self.anndata, self.genome_file, test_idx, self.chromsizes_file
            )
        elif stage == "predict":
            self.predict_dataset = AnnDataSet(
                self.anndata, self.genome_file, all_idx, self.chromsizes_file
            )
        else:
            raise ValueError("Stage must be one of 'fit', 'test', or 'predict'.")

    def train_dataloader(
        self, batch_size: int, shuffle: bool = False
    ) -> tf.data.Dataset:
        """Train dataloader"""
        if self.train_dataset is None:
            raise ValueError(
                "Train dataset is not setup. Run `setup(stage='fit') first.`"
            )
        return self._create_dataloader(self.train_dataset, batch_size, shuffle)

    def val_dataloader(self, batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
        """Val dataloader"""
        if self.val_dataset is None:
            raise ValueError(
                "Val dataset is not setup. Run `setup(stage='fit') first.`"
            )
        return self._create_dataloader(self.val_dataset, batch_size, shuffle)

    def test_dataloader(
        self, batch_size: int, shuffle: bool = False
    ) -> tf.data.Dataset:
        """Test dataloader"""
        if self.test_dataset is None:
            raise ValueError(
                "Test dataset is not setup. Run `setup(stage='test') first.`"
            )
        return self._create_dataloader(self.test_dataset, batch_size, shuffle)

    def predict_dataloader(
        self, batch_size: int, shuffle: bool = False
    ) -> tf.data.Dataset:
        """Predict dataloader"""
        if self.predict_dataset is None:
            raise ValueError(
                "Predict dataset is not setup. Run `setup(stage='predict') first.`"
            )
        return self._create_dataloader(self.predict_dataset, batch_size, shuffle)

    def info(self):
        """Get info on the current datasets."""
        datasets = {
            "n_train": self.train_dataset,
            "n_val": self.val_dataset,
            "n_test": self.test_dataset,
            "n_predict": self.predict_dataset,
        }
        datasets = {
            key: len(value) if value is not None else None
            for key, value in datasets.items()
        }
        other_info = {
            "seq_len": self.seq_len,
            "num_outputs": self.num_outputs,
        }
        return {**datasets, **other_info}

    def _create_dataloader(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        shuffle: bool,
    ):
        ds = self._generator(dataset)
        ds = ds.map(
            lambda seq, tgt: self._map_one_hot_encode(seq, tgt),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=batch_size * 8, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _generator(self, dataset) -> tf.data.Dataset:
        """Returns tf dataset generator."""
        return tf.data.Dataset.from_generator(
            dataset.__iter__,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(self.num_outputs,), dtype=tf.float32),
            ),
        )

    @tf.function
    def _map_one_hot_encode(self, sequence, target):
        """One hot encoding as a tf mapping function during prefetching."""
        if isinstance(sequence, str):
            sequence = tf.constant([sequence])
        elif isinstance(sequence, tf.Tensor) and sequence.ndim == 0:
            sequence = tf.expand_dims(sequence, 0)

        # Define one_hot_encode function using TensorFlow operations
        def one_hot_encode(sequence):
            # Map each base to an integer
            char_seq = tf.strings.unicode_split(sequence, "UTF-8")
            integer_seq = self.table.lookup(char_seq)
            # One-hot encode the integer sequence
            x = tf.one_hot(integer_seq, depth=4)
            return x

        # Apply one_hot_encode to each sequence
        one_hot_sequence = tf.map_fn(
            one_hot_encode,
            sequence,
            fn_output_signature=tf.TensorSpec(
                shape=(self.seq_len, 4), dtype=tf.float32
            ),
        )
        one_hot_sequence = tf.squeeze(one_hot_sequence, axis=0)  # remove extra map dim
        return one_hot_sequence, target


if __name__ == "__main__":
    # Test the dataloader
    # TODO: remove
    import pandas as pd
    import scipy.sparse as sp

    from enhancerai.pp import train_val_test_split

    def create_anndata_with_regions(
        regions: list[str],
        chr_var_key: str = "chr",
        compress: bool = False,
        random_state: int = None,
    ) -> AnnData:
        if random_state is not None:
            np.random.seed(random_state)
        data = np.random.randn(10, len(regions))
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

    adata = create_anndata_with_regions(regions, compress=True, random_state=42)
    train_val_test_split(
        adata,
        strategy="region",
        val_size=0.1,
        test_size=0.1,
        shuffle=True,
        random_state=42,
    )

    # Test anndataset
    # dataset = AnnDataSet(
    #     adata, genome_file, adata.var_names, random_reverse_complement=True
    # )
    # for x, y in dataset:
    #     print(x, y)

    # # test dataloader
    dataloader = AnnDataLoader(adata, genome_file)
    dataloader.setup(stage="fit", random_reverse_complement=True)
    train_loader = dataloader.train_dataloader(shuffle=True, batch_size=2)
    val_loader = dataloader.val_dataloader(shuffle=False, batch_size=4)

    for x, y in train_loader:
        print(x.shape, y.shape)
        break
