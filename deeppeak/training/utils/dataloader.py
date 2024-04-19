"""Helper functions for loading data from tfRecords."""
from __future__ import annotations
import tensorflow as tf
import os
import pyfaidx
import numpy as np
import shutil
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
    gini_std_threshold: float = 1.0,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Filter bed regions & targets based on high Gini score. Saves filtered bed regions
    back to original file and returns filtered targets.

    Args:
        target_vector (np.ndarray): targets
        bed_filename (str): path to BED file
        gini_threshold (float): Threshold for Gini scores to identify high variability.
    """

    gini_scores = calc_gini(target_vector)
    mean = np.mean(np.max(gini_scores, axis=1))
    std_dev = np.std(np.max(gini_scores, axis=1))
    gini_threshold =  mean + gini_std_threshold * std_dev
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

        gini_scores = calc_gini(
            target_vector[np.argsort(filtered_col)[::-1][:top_k_index]]
        )
        mean = np.mean(np.max(gini_scores, axis=1))
        std_dev = np.std(np.max(gini_scores, axis=1))
        gini_threshold =  mean - gini_std_threshold * std_dev
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

def write_to_bedfile(regions, output_path):
    with open(output_path, "w") as outfile:
        for region in regions:
            outfile.write("\t".join([str(x) for x in region]) + "\n")


class CustomDataset:
    def __init__(
        self,
        bed_file: str,
        genome_fasta_file: str,
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
        peak_normalization: bool = False
    ):
        # Load datasets
        self.reverse_complement = reverse_complement
        self.all_regions = self._load_bed_file(bed_file)
        self.genomic_pyfasta = pyfaidx.Fasta(
            genome_fasta_file, sequence_always_upper=True
        )
        self.targets = np.load(targets)["targets"]

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
        print(self.targets.shape)

        if(peak_normalization):
            self.targets, norm_weights = normalize_peaks(target_vector=self.targets, num_cell_types=self.targets.shape[1])
            np.savez(os.path.join(output_dir, 'normalization_weights.npz'), weights=norm_weights)

        if self.reverse_complement:
            self.targets = np.repeat(self.targets, 2, axis=0)  # double targets for each region
        
        if(specificity_filtering):
            self.targets, self.all_regions = filter_regions_on_specificity(self.targets, self.all_regions)

        self.split_dict = split_dict
        print(self.targets.shape)
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
        
        if(self.reverse_complement):
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
                train=train_indices[::2]//2 if self.reverse_complement else train_indices,
                val=val_indices//2 if self.reverse_complement else val_indices,
                test=test_indices//2 if self.reverse_complement else test_indices,
            )
            train_targets = self.targets[train_indices]
            val_targets = self.targets[val_indices]
            test_targets = self.targets[test_indices]
            np.savez(
                os.path.join(output_dir, "targets.npz"),
                train=train_targets[::2] if self.reverse_complement else train_targets,
                val=val_targets,
                test=test_targets,
            )
            regions_to_bed = self.all_regions[::2] if self.reverse_complement else self.all_regions
            write_to_bedfile(regions_to_bed, os.path.join(output_dir, "regions.bed"))
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
                    - min(self.shift_n_bp, start),                       # make sure start does not go below 0
                      min(self.shift_n_bp, self.chromsizes[chrom] - end) # make sure end does not go above chromsize
                )
                start += shift
                end += shift
            if(strand=='+'):
                sequence = str(self.genomic_pyfasta[chrom][start:end].seq)
            elif(strand == '-'):
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
            all_chromosomes = set(chrom for chrom, _, _,_ in all_regions)
            selected_chromosomes = all_chromosomes - excluded_chromosomes

        indices = [
            i
            for i, region in enumerate(all_regions)
            if region[0] in selected_chromosomes
        ]
        if fraction_of_data != 1.0:
            indices = indices[: int(np.ceil(len(indices) * fraction_of_data))]
        return np.array(indices), selected_chromosomes

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
                regions.append((chrom, start, end,'+'))
                if(self.reverse_complement):
                    regions.append((chrom, start, end, '-'))
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
