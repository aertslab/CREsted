"""Helper functions for loading data."""

from __future__ import annotations

import warnings
import tensorflow as tf
import os
from pyfaidx import Fasta
import numpy as np
import json
from typing import Tuple
from tqdm import tqdm


def _calc_gini(targets: np.ndarray) -> np.ndarray:
    """Returns gini scores for the given targets"""

    def _gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        array = (
            array.flatten().clip(0, None) + 0.0000001
        )  # Ensure non-negative values and avoid zero
        array = np.sort(array)
        index = np.arange(1, array.size + 1)
        return (np.sum((2 * index - array.size - 1) * array)) / (
            array.size * np.sum(array)
        )

    gini_scores = np.zeros_like(targets)

    for region_idx in range(targets.shape[0]):
        region_scores = targets[region_idx]
        max_idx = np.argmax(region_scores)
        gini_scores[region_idx, max_idx] = _gini(region_scores)

    return gini_scores


def filter_regions_on_specificity(
    target_vector: np.ndarray,
    regions_bed: list,
    gini_std_threshold: float = 1,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Filter bed regions & targets based on high Gini score. Saves filtered bed regions
    back to original file and returns filtered targets.

    Args:
        target_vector (np.ndarray): targets
        bed_filename (str): path to BED file
        gini_threshold (float): Threshold for Gini scores to identify high variability.
        target_idx (int): Type of targets to use for filtering decision (1='mean')
    """

    gini_scores = _calc_gini(target_vector)
    mean = np.mean(np.max(gini_scores, axis=1))
    std_dev = np.std(np.max(gini_scores, axis=1))
    gini_threshold = mean + gini_std_threshold * std_dev
    selected_indices = np.argwhere(np.max(gini_scores, axis=1) > gini_threshold)[:, 0]

    target_vector_filt = target_vector[selected_indices]
    regions_filt = [regions_bed[i] for i in selected_indices]
    print(
        f"After specificity filtering, kept {len(target_vector_filt)} out of {len(target_vector)} regions."
    )

    return target_vector_filt, regions_filt


def normalize_peaks(
    target_vector: np.ndarray,
    num_cell_types: int,
    peak_threshold: int = 0,
    gini_std_threshold: float = 1.0,
    top_k_percent: float = 0.01,
) -> np.ndarray:
    """
    Normalize the given target vector based on top peaks per cell type.

    Calculates gini scores for the top_k highest peaks. Gini scores
    below gini_threshold are considered 'low' (in variability) and are used to
    calculate weights per cell type, which are then used to normalize the targets
    accross cells types.

    Parameters:
    - target_vector (np.ndarray): The target vector to be normalized.
    - num_cell_types (int): The number of cell types in the target vector.
    - threshold (int): A threshold value for filtering the target vector.
    - gini_threshold (float): Threshold for Gini scores to identify high variability.
    - top_k_percent (float): The percentage (expressed as a fraction) of top values to
      consider for Gini score calculation.

    Returns:
    - np.ndarray: The normalized target vector with adjustments based on Gini score.
    """
    top_k_percent_means = []
    gini_scores_all = []

    print("Filtering on top k Gini scores...")
    for i in range(num_cell_types):
        filtered_col = target_vector[:, i][target_vector[:, i] > peak_threshold]
        sorted_col = np.sort(filtered_col)[::-1]
        top_k_index = int(len(sorted_col) * top_k_percent)

        gini_scores = _calc_gini(
            target_vector[np.argsort(filtered_col)[::-1][:top_k_index]]
        )
        mean = np.mean(np.max(gini_scores, axis=1))
        std_dev = np.std(np.max(gini_scores, axis=1))
        gini_threshold = mean - gini_std_threshold * std_dev
        low_gini_indices = np.where(np.max(gini_scores, axis=1) < gini_threshold)[0]
        print(f"{len(low_gini_indices)} out of {top_k_index} remain for cell type {i}.")

        if len(low_gini_indices) > 0:
            top_k_mean = np.mean(sorted_col[low_gini_indices])
            gini_scores_all.append(np.max(gini_scores[low_gini_indices], axis=1))
        else:
            top_k_mean = 0
            gini_scores_all.append(0)

        top_k_percent_means.append(top_k_mean)

    max_mean = np.max(top_k_percent_means)
    weights = max_mean / np.array(top_k_percent_means)
    print("Cell type weights:", weights)

    target_vector = target_vector * weights
    return target_vector, weights


def filter_regions_on_targets(
    target_vector: np.ndarray,
    regions_bed: list,
    targets_n_to_remove: np.array = np.array([0]),
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Filter bed regions & targets based on number of positive targets. Saves filtered bed regions
    back to original file and returns filtered targets. Mainly for deeptopic where truth is binary.

    Args:
        target_vector (np.ndarray): targets
        bed_filename (str): path to BED file
        targets_n_to_remove (tuple): Remove regions with n positive targets.
    """

    selected_indices = np.argwhere(
        np.isin(np.sum(target_vector, axis=1), targets_n_to_remove, invert=True)
    )[:, 0]

    target_vector_filt = target_vector[selected_indices]
    regions_filt = [regions_bed[i] for i in selected_indices]
    print(
        f"After deeptopic target filtering, kept {len(target_vector_filt)} out of {len(target_vector)} regions."
    )

    return target_vector_filt, regions_filt


def write_to_bedfile(regions, output_path):
    with open(output_path, "w") as outfile:
        for region in regions:
            outfile.write("\t".join([str(x) for x in region]) + "\n")


def load_targets(
    targets: str, task: str, target_goal: str, reverse_complement: bool = False
) -> np.ndarray:
    """Load the preprocessed targets from a compressed numpy file.

    Args:
        targets (str): Path to the compressed numpy file containing targets.
        task (str): The task for which the targets were created.
        target_goal (str): The target goal to use (max, mean, count, logcount).

    Returns:
        np.ndarray: Target vector with shape (regions x cell types/topics).
    """
    targets = np.load(targets)["targets"]

    # Select deeppeak targets (deep topic targets are already in correct shape)
    if task == "deeppeak":
        if targets.shape[0] == 1:
            print("Only found one target type in target array. Using that one.")
            targets = targets[0, :]
        else:
            if target_goal == "max":
                targets = targets[0, :]
            elif target_goal == "mean":
                targets = targets[1, :]
            elif target_goal == "count":
                targets = targets[2, :]
            elif target_goal == "logcount":
                targets = targets[3, :]

    if reverse_complement:
        targets = np.repeat(targets, 2, axis=0)  # double targets for each region

    return targets  # (regions x cell types/topics)


def stochastic_shift_augment(start: int, end: int, chromsize: int, shift_n_bp: int):
    """Shift augmentation for data augmentation."""
    shift = np.random.randint(
        -min(shift_n_bp, start),  # make sure start does not go below 0
        min(shift_n_bp, chromsize - end),  # make sure end does not go above chromsize
    )
    start += shift
    end += shift
    return start, end


class SequenceDataset:
    """Dataset class for loading genomic regions and targets based on split (train/val/test)."""

    def __init__(
        self,
        regions_bed_file: str,
        genome_fasta_file: str,
        targets_file: str,
        config: dict,
        chromsizes: dict[str, int] | None = None,
        output_dir: str | None = None,
    ):
        self.num_classes = int(config["num_classes"])
        self.chromsizes = chromsizes
        self.shift_n_bp = int(config["augment_shift_n_bp"])

        if self.chromsizes is None and self.shift_n_bp > 0:
            raise ValueError(
                "Chromsizes must be provided for stochastic shift augmentation."
            )

        self.regionloader = GenomicRegionLoader(
            regions_bed_file,
            genome_fasta_file,
            config["rev_complement"],
            int(config["augment_shift_n_bp"]),
            self.chromsizes,
        )
        self.targets = load_targets(
            targets_file,
            config["task"],
            config["deeppeak"]["target"],
            config["rev_complement"],
        )

        assert len(self.targets) == len(
            self.regionloader.regions
        ), f"""Target vector and regions 
        are not the same length,
        please make sure that you are using the correct inputs, target vector length: {len(self.targets)},
        regions length: {len(self.regionloader.regions)}"""

        if config["peak_normalization"]:
            print("Normalizing peaks...")
            self.targets, norm_weights = normalize_peaks(
                target_vector=self.targets, num_cell_types=self.targets.shape[1]
            )

        if config["specificity_filtering"]:
            print("Filtering regions based on region specificity...")
            self.targets, self.regionloader.regions = filter_regions_on_specificity(
                self.targets, self.regionloader.regions
            )

        if config["task"] == "deeptopic":
            print("Filtering regions with zero targets...")
            self.targets, self.regionloader.regions = filter_regions_on_targets(
                self.targets, self.regionloader.regions
            )

        # Get split indices
        self.splitter = DatasetSplitter(self.regionloader.regions)
        self.indices = self.splitter.split_datasets(config["split"])  # dict of lists

        if float(config["fraction_of_data"]) != 1.0:
            # debugging purposes
            for subset in self.indices:
                self.indices[subset] = self.indices[subset][
                    : int(
                        np.ceil(
                            len(self.indices[subset])
                            * float(config["fraction_of_data"])
                        )
                    )
                ]

        # Remove augmented regions from val and test set if augmented in preprocessing
        if config["shift_augmentation"]["use"]:
            n_shifts = int(config["shift_augmentation"]["n_shifts"])
            for subset in ["val", "test"]:
                self.indices[subset] = [
                    i for i in self.indices[subset] if i % (n_shifts * 2 + 1) == 0
                ]

        # Remove reverse complement regions if reverse complement is used
        if config["rev_complement"]:
            for subset in ["val", "test"]:
                self.indices[subset] = self.indices[subset][::2]

        # Save outputs
        if output_dir is not None:
            self.save_outputs(
                output_dir, self.splitter.split_dict, config["rev_complement"]
            )
            if config["peak_normalization"]:
                np.savez(
                    os.path.join(output_dir, "normalization_weights.npz"),
                    weights=norm_weights,
                )

        # shuffle train indices
        if config["shuffle"]:
            np.random.shuffle(self.indices["train"])

    def len(self, subset: str):
        """Return the number of samples in the given split."""
        return len(self.indices[subset])

    def generator(self, split: str | bytes):
        """Yield sequences and targets for the given split."""
        split = split.decode() if isinstance(split, bytes) else split
        for sample_idx in self.indices[split]:
            region = self.regionloader.get_region(sample_idx)
            chrom, start, end, strand = region
            if self.shift_n_bp > 0:
                # stochastic shifting
                if split == "train":
                    shift_n = np.random.randint(-self.shift_n_bp, self.shift_n_bp)
                else:
                    shift_n = 0
                seq_start = self.shift_n_bp + shift_n
                seq_end = (end - start) + seq_start
            sequence = str(self.regionloader.get_sequence(region))
            if self.shift_n_bp > 0:
                sequence = sequence[seq_start:seq_end]
            target = self.targets[sample_idx]
            yield sequence, target

    def subset(self, split: str):
        """Returns dataset generator for the given split."""
        return tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32),
            ),
            args=(split,),
        )

    def save_outputs(
        self, output_dir: str, split_dict: dict, reverse_complement: bool = False
    ):
        """Save the split indices and targets to the output directory."""
        with open(os.path.join(output_dir, "chrom_mapping.json"), "w") as f:
            json.dump(split_dict, f)
        indices_train = (
            self.indices["train"][::2] if reverse_complement else self.indices["train"]
        )
        indices_val = (
            self.indices["val"][::2] if reverse_complement else self.indices["val"]
        )
        indices_test = (
            self.indices["test"][::2] if reverse_complement else self.indices["test"]
        )

        np.savez(
            os.path.join(output_dir, "region_split_ids.npz"),
            train=indices_train,
            val=indices_val,
            test=indices_test,
        )

        train_targets = self.targets[indices_train]
        val_targets = self.targets[indices_val]
        test_targets = self.targets[indices_test]
        np.savez(
            os.path.join(output_dir, "targets.npz"),
            train=train_targets,
            val=val_targets,
            test=test_targets,
        )

        write_to_bedfile(
            (
                self.regionloader.regions[::2]
                if reverse_complement
                else self.regionloader.regions
            ),
            os.path.join(output_dir, "regions.bed"),
        )
        print(f"Saved bed regions, split ids and split targets to {output_dir}")


class GenomicRegionLoader:
    """Handles loading of genomic regions and sequences from a BED & Fasta file."""

    def __init__(
        self,
        bed_file: str,
        genome_fasta_file: str,
        reverse_complement: bool = False,
        max_stochastic_shift: int = 0,
        chromsizes: dict[str, int] | None = None,
    ):
        self.regions = self.load_bed_file(bed_file, reverse_complement)
        self.genome = self.load_genomic_data(
            genome_fasta_file, self.regions, max_stochastic_shift, chromsizes
        )

    def load_genomic_data(
        self,
        genome_fasta_file: str,
        regions: list,
        max_stochastic_shift: int,
        chromsizes: dict[str, int] | None = None,
    ) -> dict[str, str]:
        """
        Load genomic data into memory for faster access and to be able to fully shuffle
        the data without running into Fasta binning issues. Load some extra letters into
        memory per region if stochastic shifting.
        """
        genome_dict = {}
        genome = Fasta(genome_fasta_file, sequence_always_upper=True)
        for region in tqdm(regions, desc="Loading sequences into memory..."):
            if region not in genome_dict:
                chrom, start, end, strand = region
                if max_stochastic_shift > 0:
                    # Loading wider regions if stochastic shifting. Corrected in the generator.
                    start -= max_stochastic_shift
                    end += max_stochastic_shift
                    chromsize = chromsizes[chrom]
                    if start < 0:
                        end = abs(start) + end
                        start = 0
                    if end > chromsize:
                        start = start - (end - chromsize)
                        end = chromsize
                if strand == "+":
                    genome_dict[region] = genome[chrom][start:end].seq
                elif strand == "-":
                    genome_dict[region] = genome[chrom][
                        start:end
                    ].reverse.complement.seq
        return genome_dict

    def load_bed_file(self, bed_file: str, reverse_complement: bool = False) -> list:
        regions = []
        with open(bed_file, "r") as fh_bed:
            for line in fh_bed:
                line = line.rstrip("\r\n")
                if line.startswith("#"):
                    continue
                columns = line.split("\t")
                chrom = columns[0]
                start, end = [int(x) for x in columns[1:3]]
                regions.append((chrom, start, end, "+"))
                if reverse_complement:
                    regions.append((chrom, start, end, "-"))
        return regions

    def get_region(self, idx):
        return self.regions[idx]

    def get_sequence(self, region):
        return self.genome[region]


class DatasetSplitter:
    """Handles splitting of dataset into train, val, test sets."""

    def __init__(self, regions):
        self.regions = regions
        self.indices = {"train": [], "val": [], "test": []}
        self.split_dict = {}

    def split_datasets(self, split_config: dict) -> dict[str, list[int]]:
        """
        Split the dataset into train, val, test sets using given strategy.

        Args:
            split_config (dict): Dictionary containing keys: 'strategy', 'val_chroms', 'test_chroms', 'train_fraction', 'val_fraction'

        Returns:
            dict: Dictionary containing indices for train, val, test sets.
        """
        self.split_dict = split_config
        if split_config["strategy"] == "chr":
            self._split_by_chromosome(
                val_chroms=split_config["val_chroms"],
                test_chroms=split_config["test_chroms"],
            )
        elif split_config["strategy"] == "region":
            self._split_regions(
                val_fraction=split_config["val_fraction"],
                test_fraction=split_config["test_fraction"],
            )
        elif split_config["strategy"] == "chr_auto":
            self._split_by_chromosome_auto(
                val_fraction=split_config["val_fraction"],
                test_fraction=split_config["test_fraction"],
            )
        else:
            raise ValueError(
                "Invalid splitting strategy. Choose 'chr', 'random', or 'chr_auto'"
            )
        return self.indices

    def _split_by_chromosome(self, val_chroms: list[str], test_chroms: list[str]):
        """Split the dataset based on selected chrosomomes. If the same chromosomes are supplied
        to both val and test, then the regions while be divided evenly between.

        Args:
            val_chroms (list): List of chromosome names to include in val set.
            test_chroms (list): List of chromosome names to include in test set.
        """
        print("Using selected chromosomes for val and test sets...")
        # Check data correctness
        if len(val_chroms) == 0:
            warnings.warn(
                "No chromosomes specified for val set. Val set will be empty."
            )
        if len(test_chroms) == 0:
            warnings.warn(
                "No chromosomes specified for test set. Test set will be empty."
            )

        all_chroms = set(chrom for chrom, _, _, _ in self.regions)
        if not set(val_chroms) <= all_chroms:
            raise ValueError("One or more val chromosomes not found in regions.")
        if not set(test_chroms) <= all_chroms:
            raise ValueError("One or more test chromosomes not found in regions.")

        # Split
        overlap_chroms = set(val_chroms) & set(test_chroms)
        val_chroms = set(val_chroms) - overlap_chroms
        test_chroms = set(test_chroms) - overlap_chroms
        chrom_counter = {}  # Used if same chroms in val and test

        for i, region in enumerate(self.regions):
            chrom = region[0]
            if chrom in overlap_chroms:
                if chrom not in chrom_counter:
                    chrom_counter[chrom] = 0
                # Alternate between validation and test for overlapping chromosomes
                if chrom_counter[chrom] % 2 == 0:
                    self.indices["val"].append(i)
                else:
                    self.indices["test"].append(i)
                chrom_counter[chrom] += 1
            elif chrom in val_chroms:
                self.indices["val"].append(i)
            elif chrom in test_chroms:
                self.indices["test"].append(i)
            else:
                self.indices["train"].append(i)
        print(
            f"Val chromosomes: {val_chroms.union(overlap_chroms)}, fraction: {len(self.indices['val'])/len(self.regions):.2f}"
        )
        print(
            f"Test chromosomes: {test_chroms.union(overlap_chroms)}, fraction: {len(self.indices['test'])/len(self.regions):.2f}"
        )

    def _split_by_chromosome_auto(
        self, val_fraction: float = 0.1, test_fraction: float = 0.1
    ):
        """Split the dataset based on chromosome, automatically selecting chromosomes for val and test sets.

        Args:
            val_fraction (float): Fraction of regions to include in val set.
            test_fraction (float): Fraction of regions to include in test set.
        """
        from collections import defaultdict

        print("Auto-splitting on chromosomes...")

        # Count regions per chromosome
        chrom_count = defaultdict(int)
        for region in self.regions:
            chrom_count[region[0]] += 1

        total_regions = sum(chrom_count.values())
        target_val_size = int(val_fraction * total_regions)  # 10% each for val and test
        target_test_size = int(test_fraction * total_regions)

        # Shuffle chromosomes to randomize selection
        chromosomes = list(chrom_count.keys())

        val_chroms = set()
        test_chroms = set()
        current_val_size = 0
        current_test_size = 0

        # Assign chromosomes to val and test sets
        for chrom in chromosomes:
            if current_val_size < target_val_size:
                val_chroms.add(chrom)
                current_val_size += chrom_count[chrom]
            elif current_test_size < target_test_size:
                test_chroms.add(chrom)
                current_test_size += chrom_count[chrom]
            if (
                current_val_size >= target_val_size
                and current_test_size >= target_test_size
            ):
                break

        # Assign indices to train, val, test based on the selected chromosomes
        for i, region in enumerate(self.regions):
            if region[0] in val_chroms:
                self.indices["val"].append(i)
            elif region[0] in test_chroms:
                self.indices["test"].append(i)
            else:
                self.indices["train"].append(i)

        print(
            f"Val chromosomes: {val_chroms}, fraction: {current_val_size/total_regions:.2f}"
        )
        print(
            f"Test chromosomes: {test_chroms}, fraction: {current_test_size/total_regions:.2f}"
        )
        self.split_dict["val_chroms"] = list(val_chroms)
        self.split_dict["test_chroms"] = list(test_chroms)

    def _split_regions(self, val_fraction: float = 0.1, test_fraction: float = 0.1):
        """Split into train, val, test sets based on regions.

        Args:
            val_fraction (float): Fraction of regions to include in val set.
            test_fraction (float): Fraction of regions to include in test set.
        """
        print(f"Splitting on regions with fractions: {val_fraction}, {test_fraction}")
        total_indices = np.arange(len(self.regions))
        num_val = int(len(total_indices) * val_fraction)
        num_test = int(len(total_indices) * test_fraction)
        num_train = len(total_indices) - num_val - num_test

        self.indices["train"] = total_indices[:num_train]
        self.indices["val"] = total_indices[num_train : num_train + num_val]
        self.indices["test"] = total_indices[num_train + num_val :]
        print("done")

    def get_indices(self, subset):
        return self.indices.get(subset, [])

    def get_subset(self, subset):
        return [self.regions[idx] for idx in self.indices.get(subset, [])]


if __name__ == "__main__":
    # test dataloader
    bed_file = "data/processed/consensus_peaks_inputs.bed"
    genome_fasta_file = "data/raw/genome.fa"
    targets = "data/processed/targets_deeptopic.npz"
    chromsizes = "data/raw/chrom.sizes"

    config = {
        "num_classes": 80,
        "split": {
            "strategy": "chr_auto",
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "val_chroms": ["chr10"],
            "test_chroms": [],
        },
        "rev_complement": True,
        "specificity_filtering": False,
        "augment_shift_n_bp": 100,
        "task": "deeptopic",
        "deeppeak": {"target": "mean"},
        "fraction_of_data": 1.0,
        "shift_augmentation": {"use": False, "n_shifts": 2},
        "peak_normalization": False,
        "shuffle": True,
    }

    def _load_chromsizes(chrom_sizes_file: str) -> dict[str, int]:
        chrom_sizes = {}
        with open(chrom_sizes_file, "r") as sizes:
            for line in sizes:
                chrom, s_size = line.strip().split("\t")[0:2]
                i_size = int(s_size)
                chrom_sizes[chrom] = i_size
        return chrom_sizes

    dataset = SequenceDataset(
        bed_file, genome_fasta_file, targets, config, _load_chromsizes(chromsizes)
    )

    seq_len = 500

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
