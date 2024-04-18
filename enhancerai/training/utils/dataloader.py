"""Helper functions for loading data from tfRecords."""

from __future__ import annotations
import tensorflow as tf
import os
from pyfaidx import Fasta
import numpy as np
import json
from typing import Tuple


def calc_gini(targets: np.ndarray) -> np.ndarray:
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

    gini_scores = calc_gini(target_vector)
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


def write_to_bedfile(regions, output_path):
    with open(output_path, "w") as outfile:
        for region in regions:
            outfile.write("\t".join([str(x) for x in region]) + "\n")


class CustomDataset:
    def __init__(
        self,
        bed_file: str,
        genome_fasta_file: str,
        task: str,
        targets: str,
        target_goal: str,
        split_dict: dict,
        num_classes: int,
        shift_n_bp: int = 0,
        fraction_of_data: float = 1.0,
        output_dir: str | None = None,
        chromsizes: dict[str, int] | None = None,
        reverse_complement: bool = False,
        specificity_filtering: bool = False,
        shift_augmentation_pre_used: bool = False,
        shift_augmentation_pre_used_n_shifts: int = 2,
    ):
        # Load datasets
        self.reverse_complement = reverse_complement
        self.all_regions = self._load_bed_file(bed_file)
        self.genomic_pyfasta = Fasta(genome_fasta_file, sequence_always_upper=True)
        self.targets = np.load(targets)["targets"]

        if task == "deeppeak":
            if self.targets.shape[0] == 1:
                print("Only found one target type in target array. Using that one.")
                self.targets = self.targets[0, :]
            else:
                if target_goal == "max":
                    self.targets = self.targets[0, :]
                elif target_goal == "mean":
                    self.targets = self.targets[1, :]
                elif target_goal == "count":
                    self.targets = self.targets[2, :]
                elif target_goal == "logcount":
                    self.targets = self.targets[3, :]

            if specificity_filtering:
                self.targets, self.all_regions = filter_regions_on_specificity(
                    self.targets, self.all_regions
                )
        # if task is deeptopic, then already in correct shape

        if self.reverse_complement:
            self.targets = np.repeat(
                self.targets, 2, axis=0
            )  # double targets for each region
        self.split_dict = split_dict

        self.num_classes = num_classes
        self.shift_n_bp = shift_n_bp
        if chromsizes is None:
            # chromsizes must be provided for shift augmentation
            if shift_n_bp > 0:
                raise ValueError("Chromsizes must be provided for shift augmentation.")
        else:
            # Check wether all chromosomes are present in the chromsizes dict
            chroms = set(chrom for chrom, _, _, _ in self.all_regions)
            chroms_not_in_chromsizes = chroms - set(chromsizes.keys())
            if len(chroms_not_in_chromsizes) > 0:
                raise ValueError(
                    f"Chromsizes dict does not contain all chromosomes in the bed file.\nMissing: {', '.join(chroms_not_in_chromsizes)}"
                )
        self.chromsizes = chromsizes

        # Get indices for each set type
        val_indices, val_chroms = self._get_indices_for_set_type(
            self.split_dict, "val", self.all_regions, fraction_of_data
        )
        test_indices, test_chroms = self._get_indices_for_set_type(
            self.split_dict, "test", self.all_regions, fraction_of_data
        )
        train_indices, train_chroms = self._get_indices_for_set_type(
            self.split_dict, "train", self.all_regions, fraction_of_data
        )

        # Remove augmented regions from val and test set if augmented in preprocessing
        if shift_augmentation_pre_used:
            n_shifts = shift_augmentation_pre_used_n_shifts
            val_indices = [i for i in val_indices if i % (n_shifts * 2 + 1) == 0]
            test_indices = [i for i in test_indices if i % (n_shifts * 2 + 1) == 0]

        if self.reverse_complement:
            val_indices = val_indices[::2]
            test_indices = test_indices[::2]

        self.indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
        self.chroms = {
            "train": list(train_chroms),
            "val": list(val_chroms),
            "test": list(test_chroms),
        }
        if output_dir is not None:
            # Save chromosome mapping to output directory for safekeeping.
            with open(os.path.join(output_dir, "chrom_mapping.json"), "w") as f:
                json.dump(self.chroms, f)

            # Save split ids & targets to output directory for safekeeping.
            np.savez(
                os.path.join(output_dir, "region_split_ids.npz"),
                train=train_indices,
                val=val_indices,
                test=test_indices,
            )
            train_targets = self.targets[train_indices]
            val_targets = self.targets[val_indices]
            test_targets = self.targets[test_indices]
            np.savez(
                os.path.join(output_dir, "targets.npz"),
                train=train_targets,
                val=val_targets,
                test=test_targets,
            )
            write_to_bedfile(self.all_regions, os.path.join(output_dir, "regions.bed"))
            print(f"Saved bed regions, split ids and split targets to {output_dir}")

    def len(self, subset: str):
        if subset not in ["train", "val"]:
            raise ValueError("subset must be 'train' or 'val'")
        return len(self.indices[subset])

    def generator(self, split: str | bytes):
        split = split.decode() if isinstance(split, bytes) else split
        for sample_idx in self.indices[split]:
            region = self.all_regions[sample_idx]
            chrom, start, end, strand = region
            if (self.shift_n_bp > 0) and (split == "train"):
                # shift augmentation
                shift = np.random.randint(
                    -min(self.shift_n_bp, start),  # make sure start does not go below 0
                    min(
                        self.shift_n_bp, self.chromsizes[chrom] - end
                    ),  # make sure end does not go above chromsize
                )
                start += shift
                end += shift
            if strand == "+":
                sequence = str(self.genomic_pyfasta[chrom][start:end].seq)
            elif strand == "-":
                sequence = str(self.genomic_pyfasta[chrom][start:end].complement.seq)
            else:
                raise ValueError("Strand must be '+' or '-'")
            target = self.targets[sample_idx]
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

        if set_type in ["val", "test"]:
            # If no splits given, then auto split, trying to achieve target fraction
            if len(split_dict.get(set_type)) == 0:
                print(f"Auto-splitting {set_type} set...")
                # Determine chromosomes already used in other sets
                other_sets = {"train", "val", "test"} - {set_type}
                used_chromosomes = set()
                for other_set in other_sets:
                    used_chromosomes.update(split_dict.get(other_set, []))

                # Select chromosomes for the current set
                selected_chromosomes = self._select_chromosomes_for_subset(
                    used_chromosomes, set_type, target_fraction=0.1
                )
                self.split_dict[set_type] = list(selected_chromosomes)
            if len(split_dict.get(set_type)) > 0:
                selected_chromosomes = set(split_dict.get(set_type, []))

        elif set_type == "train":
            excluded_chromosomes = set(
                split_dict.get("val", []) + split_dict.get("test", [])
            )
            all_chromosomes = set(chrom for chrom, _, _, _ in all_regions)
            selected_chromosomes = all_chromosomes - excluded_chromosomes

        indices = [
            i
            for i, region in enumerate(all_regions)
            if region[0] in selected_chromosomes
        ]
        if fraction_of_data != 1.0:
            indices = indices[: int(np.ceil(len(indices) * fraction_of_data))]
        return indices, selected_chromosomes

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
                regions.append((chrom, start, end, "+"))
                if self.reverse_complement:
                    regions.append((chrom, start, end, "-"))
        return regions

    def _get_chromosome_counts(self):
        """Count occurrences of each chromosome."""
        chromosome_counts = {}
        for chrom, _, _, _ in self.all_regions:
            if chrom not in chromosome_counts:
                chromosome_counts[chrom] = 0
            chromosome_counts[chrom] += 1
        return chromosome_counts

    def _select_chromosomes_for_subset(
        self, excluded_chromosomes, split, target_fraction=0.1
    ):
        """
        Auto select chromosomes for a subset (val/test) based on fraction,
        excluding those already in other subsets.
        """
        chrom_counts = self._get_chromosome_counts()
        total_seqs = sum(
            count
            for chrom, count in chrom_counts.items()
            if chrom not in excluded_chromosomes
        )
        target_count = int(round(target_fraction * total_seqs))

        # Filter out excluded chromosomes and sort the rest by count
        sorted_chroms = sorted(
            (chrom for chrom in chrom_counts if chrom not in excluded_chromosomes),
            key=chrom_counts.get,
            reverse=False,
        )

        selected_chroms, current_count = set(), 0
        for chrom in sorted_chroms:
            chrom_count = chrom_counts[chrom]
            if current_count + chrom_count <= target_count:
                selected_chroms.add(chrom)
                current_count += chrom_count
            else:
                # Check if adding this chromosome gets us closer to the target
                if abs(target_count - (current_count + chrom_count)) < abs(
                    target_count - current_count
                ):
                    selected_chroms.add(chrom)
                    current_count += chrom_count
                print(f"{split}: Using {(current_count/total_seqs)*100:.3f}% of data")
                break

        return selected_chroms


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
    def __init__(
        self,
        config: dict,
        regions_bed_file: str,
        genome_fasta_file: str,
        chromsizes: dict[str, int] | None = None,
        output_dir: str | None = None,
    ):
        self.regionloader = GenomicRegionLoader(
            regions_bed_file,
            genome_fasta_file,
            config["reverse_complement"],
        )
        self.targets = load_targets(
            config["targets"],
            config["task"],
            config["target_goal"],
            config["reverse_complement"],
        )
        self.num_classes = int(config["num_classes"])
        self.chromsizes = chromsizes
        self.shift_n_bp = int(config["augment_shift_n_bp"])

        if self.chromsizes is None and self.shift_n_bp > 0:
            raise ValueError(
                "Chromsizes must be provided for stochastic shift augmentation."
            )

        if config["specificity_filtering"]:
            self.targets, self.regionloader.regions = filter_regions_on_specificity(
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
        if config["reverse_complement"]:
            for subset in ["val", "test"]:
                self.indices[subset] = self.indices[subset][::2]

        # Save outputs
        if output_dir is not None:
            self.save_outputs(output_dir, self.splitter.split_dict)

    def __len__(self, subset: str):
        """Return the number of samples in the given split."""
        return len(self.indices[subset])

    def generator(self, split: str | bytes):
        """Yield sequences and targets for the given split."""
        split = split.decode() if isinstance(split, bytes) else split
        for sample_idx in self.indices[split]:
            region = self.regionloader.get_region(sample_idx)
            chrom, start, end, strand = region
            if split == "train" and self.shift_n_bp > 0:
                start, end = stochastic_shift_augment(
                    start, end, self.chromsizes[chrom], self.shift_n_bp
                )
            sequence = str(self.regionloader.get_sequence(chrom, start, end, strand))
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

    def save_outputs(self, output_dir: str, split_dict: dict):
        """Save the split indices and targets to the output directory."""
        with open(os.path.join(output_dir, "chrom_mapping.json"), "w") as f:
            json.dump(split_dict, f)

        np.savez(
            os.path.join(output_dir, "region_split_ids.npz"),
            train=self.indices["train"],
            val=self.indices["val"],
            test=self.indices["test"],
        )

        train_targets = self.targets[self.indices["train"]]
        val_targets = self.targets[self.indices["val"]]
        test_targets = self.targets[self.indices["test"]]
        np.savez(
            os.path.join(output_dir, "targets.npz"),
            train=train_targets,
            val=val_targets,
            test=test_targets,
        )

        write_to_bedfile(
            self.regionloader.regions, os.path.join(output_dir, "regions.bed")
        )
        print(f"Saved bed regions, split ids and split targets to {output_dir}")


class GenomicRegionLoader:
    """Handles loading of genomic regions from a BED file."""

    def __init__(
        self, bed_file: str, genome_fasta_file: str, reverse_complement: bool = False
    ):
        self.regions = self.load_bed_file(bed_file, reverse_complement)
        self.genome = self.load_genomic_data(genome_fasta_file)

    def load_genomic_data(self):
        return Fasta(self.genome_fasta_file, sequence_always_upper=True)

    def load_bed_file(self, reverse_complement: bool = False) -> list:
        regions = []
        with open(self.bed_file, "r") as fh_bed:
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

    def get_sequence(self, chrom, start, end, strand):
        if strand == "+":
            return self.genome[chrom][start:end].seq
        elif strand == "-":
            return self.genome[chrom][start:end].complement.seq
        else:
            raise ValueError("Strand must be '+' or '-'")


class DatasetSplitter:
    """Handles splitting of dataset into train, val, test sets."""

    def __init__(self, regions):
        self.regions = regions
        self.indices = {"train": [], "val": [], "test": []}
        self.split_dict = {}

    def split_datasets(self, split_config: dict):
        """
        Split the dataset into train, val, test sets using given strategy.

        Args:
            split_config (dict): Dictionary containing keys: 'strategy', 'val_chroms', 'test_chroms', 'train_fraction', 'val_fraction'
        """
        self.split_dict = split_config
        if split_config["strategy"] == "chr":
            self._split_by_chromosome(
                val_chroms=self.split_config["val_chroms"],
                test_chroms=self.split_config["test_chroms"],
            )
        elif split_config["strategy"] == "random":
            self._split_randomly(
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

    def _split_by_chromosome(
        self, val_chroms: list[str], test_chroms: list[str]
    ) -> dict[str, list[int]]:
        """Split the dataset based on selected chrosomomes. If the same chromosomes are supplied
        to both val and test, then the regions while be divided evenly between.

        Args:
            val_chroms (list): List of chromosome names to include in val set.
            test_chroms (list): List of chromosome names to include in test set.
        """
        print("Using selected chromosomes for val and test sets...")
        overlap_chroms = set(val_chroms) & set(test_chroms)
        val_chroms = set(val_chroms) - overlap_chroms
        test_chroms = set(test_chroms) - overlap_chroms
        chrom_counter = {}  # Used if same chroms in val and test

        if (len(val_chroms)) == 0 or (len(test_chroms) == 0):
            raise ValueError(
                "No chromosomes specified for val or test set. Select chromosomes or choose a different splitting strategy."
            )

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

        return self.indices

    def _split_by_chromosome_auto(
        self, val_fraction: float = 0.1, test_fraction: float = 0.1
    ) -> dict[str, list[int]]:
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
        np.random.shuffle(chromosomes)

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
        return self.indices

    def _split_randomly(
        self, val_fraction: float = 0.1, test_fraction: float = 0.1
    ) -> dict[str, list[int]]:
        """Split the regions randomly into train, val, test sets.

        Args:
            val_fraction (float): Fraction of regions to include in val set.
            test_fraction (float): Fraction of regions to include in test set.
        """
        print(
            f"Splitting randomly on regions with fractions: {val_fraction}, {test_fraction}"
        )
        total_indices = np.arange(len(self.regions))
        np.random.shuffle(total_indices)
        num_val = int(len(total_indices) * val_fraction)
        num_test = int(len(total_indices) * test_fraction)
        num_train = len(total_indices) - num_val - num_test

        self.indices["train"] = total_indices[:num_train]
        self.indices["val"] = total_indices[num_train : num_train + num_val]
        self.indices["test"] = total_indices[num_train + num_val :]
        return self.indices

    def get_indices(self, subset):
        return self.indices.get(subset, [])

    def get_subset(self, subset):
        return [self.regions[idx] for idx in self.indices.get(subset, [])]


if __name__ == "__main__":
    # test dataloader
    bed_file = "data/processed/consensus_peaks_inputs.bed"
    genome_fasta_file = "data/raw/genome.fa"
    targets = "data/processed/targets.npz"

    # split_dict = {"val": ["chr8", "chr10"], "test": ["chr9", "chr18"]}
    split_dict = {"test": ["chr1"], "val": []}

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
