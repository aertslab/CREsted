"""Helper functions for loading data from tfRecords."""
import tensorflow as tf
import os
import pyfaidx
import numpy as np


class CustomDataset:
    def __init__(
        self,
        bed_file: str,
        genome_fasta_file: str,
        targets: str,
        split_dict: dict,
        num_classes: int,
        shift_n_bp: int = 0,
        fraction_of_data: float = 1.0,
        output_dir: str = None,
    ):
        # Load datasets
        self.all_regions = self._load_bed_file(bed_file)
        self.genomic_pyfasta = pyfaidx.Fasta(
            genome_fasta_file, sequence_always_upper=True
        )
        self.targets = np.load(targets)["targets"]
        self.split_dict = split_dict

        self.num_classes = num_classes
        self.shift_n_bp = shift_n_bp

        # Get indices for each set type
        train_indices = self._get_indices_for_set_type(
            self.split_dict, "train", self.all_regions, fraction_of_data
        )
        val_indices = self._get_indices_for_set_type(
            self.split_dict, "val", self.all_regions, fraction_of_data
        )
        test_indices = self._get_indices_for_set_type(
            self.split_dict, "test", self.all_regions, fraction_of_data
        )
        self.indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

        # Save split ids & targets to output directory for safekeeping.
        if output_dir is not None:
            np.savez(
                os.path.join(output_dir, "region_split_ids.npz"),
                train=train_indices,
                val=val_indices,
                test=test_indices,
            )
            train_targets = self.targets[:, train_indices]
            val_targets = self.targets[:, val_indices]
            test_targets = self.targets[:, test_indices]
            np.savez(
                os.path.join(output_dir, "targets.npz"),
                train=train_targets,
                val=val_targets,
                test=test_targets,
            )
            print(f"Saved split ids and targets to {output_dir}")

    def len(self, subset: str):
        if subset not in ["train", "val"]:
            raise ValueError("subset must be 'train' or 'val'")
        return len(self.indices[subset])

    def generator(self, split: str):
        split = split.decode() if isinstance(split, bytes) else split
        for sample_idx in self.indices[split]:
            region = self.all_regions[sample_idx]
            chrom, start, end = region
            if (self.shift_n_bp > 0) and (split == "train"):
                # shift augmentation
                shift = np.random.randint(-self.shift_n_bp, self.shift_n_bp)
                start += shift
                end += shift
            sequence = str(self.genomic_pyfasta[chrom][start:end].seq)
            target = self.targets[1, sample_idx]
            yield sequence, target

    def subset(self, split: str):
        return tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32),
            ),
            args=(split,),
        )

    def _get_indices_for_set_type(
        self, split_dict: dict, set_type: str, all_regions: list, fraction_of_data=1.0
    ):
        """Get indices for the specified set type (train/val/test)."""
        if set_type not in ["train", "val", "test"]:
            raise ValueError("set_type must be 'train', 'val', or 'test'")

        if set_type == "train":
            excluded_chromosomes = set(
                split_dict.get("val", []) + split_dict.get("test", [])
            )
            all_chromosomes = set(chrom for chrom, _, _ in all_regions)
            selected_chromosomes = all_chromosomes - excluded_chromosomes
        else:
            selected_chromosomes = set(split_dict.get(set_type, []))

        indices = [
            i
            for i, region in enumerate(all_regions)
            if region[0] in selected_chromosomes
        ]
        if fraction_of_data != 1.0:
            indices = indices[: int(np.ceil(len(indices) * fraction_of_data))]
        return indices

    def _load_bed_file(self, regions_bed_filename: str):
        """
        Read BED file and yield a region (chrom, start, end) for each invocation.
        """
        regions = []
        with open(regions_bed_filename, "r") as fh_bed:
            for line in fh_bed:
                line = line.rstrip("\r\n")

                if line.startswith("#"):
                    continue

                columns = line.split("\t")
                chrom = columns[0]
                start, end = [int(x) for x in columns[1:3]]
                regions.append((chrom, start, end))
        return regions


if __name__ == "__main__":
    # test dataloader
    bed_file = "data/interim/consensus_peaks_2114.bed"
    genome_fasta_file = "data/raw/genome.fa"
    targets = "data/interim/targets.npy"
    targets_numpy = "data/interim/targets.npy"

    split_dict = {"val": ["chr8", "chr10"], "test": ["chr9", "chr18"]}

    dataset = CustomDataset(bed_file, genome_fasta_file, targets, split_dict, 19)

    seq_len = 2114

    base_to_int_mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(base_to_int_mapping.keys())),
            values=tf.constant(list(base_to_int_mapping.values()), dtype=tf.int32),
        ),
        default_value=-1,
    )

    def mapped_function(sequence, target):
        if isinstance(sequence, str):
            sequence = tf.constant([sequence])
        elif isinstance(sequence, tf.Tensor) and sequence.ndim == 0:
            sequence = tf.expand_dims(sequence, 0)

        # Define one_hot_encode function using TensorFlow operations
        def one_hot_encode(sequence):
            # Map each base to an integer
            char_seq = tf.strings.unicode_split(sequence, "UTF-8")
            integer_seq = table.lookup(char_seq)
            # One-hot encode the integer sequence
            return tf.one_hot(integer_seq, depth=4)

        # Apply one_hot_encode to each sequence
        one_hot_sequence = tf.map_fn(
            one_hot_encode,
            sequence,
            fn_output_signature=tf.TensorSpec(shape=(seq_len, 4), dtype=tf.float32),
        )
        one_hot_sequence = tf.squeeze(one_hot_sequence, axis=0)  # remove extra map dim
        return one_hot_sequence, target

    train_dataset = (
        dataset.subset("train")
        .map(mapped_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(1)
    )

    # Time performance
    import time

    start_time = time.perf_counter()
    i = 0
    for seq, target in train_dataset:
        i += 1
        if i == 10000:
            print(seq)
            print(target)
            break
    print("Execution time:", time.perf_counter() - start_time)
    print("Time per sample:", (time.perf_counter() - start_time) / 10000)
