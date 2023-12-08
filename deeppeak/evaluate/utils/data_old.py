"""Utils for loading and processing data for evaluating."""

import math
import glob
import pyfaidx
import pyBigWig
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial


def extend_sequence(start, end, extend_length=2114):
    """
    Extend the given sequence to a specific length.

    Args:
        start (int): Start position of the sequence.
        end (int): End position of the sequence.
        extend_length (int, optional): Length to extend the sequence.

    Returns:
        tuple: Updated start and end positions
    """
    diff = extend_length - (end - start)
    start = start - int(math.ceil(diff / 2))
    end = end + int(math.floor(diff / 2))

    return start, end


def filter_edge_regions(peaks_df, bw):
    """
    Filter regions in bed file that are on edges.

    Excludes regions that cannot be used to construct
    input length + extend length of the sequence.

    Args:
        peaks_df (pd.DataFrame): DataFrame containing peak regions.
        bw: PyBigWig object.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    total_peaks = peaks_df.shape[0]

    # left edge case
    left_edge_cond = peaks_df["start_ext"] < 0
    peaks_df = peaks_df[~(left_edge_cond)]
    num_filtered = left_edge_cond.sum()

    # right edge case
    chrom_to_sizes = bw.chroms()
    chrom_sizes_df = peaks_df["chrom"].map(chrom_to_sizes).fillna(value=0)
    right_edge_cond = peaks_df["end_ext"] > chrom_sizes_df
    peaks_df = peaks_df[~(right_edge_cond)]
    num_filtered += right_edge_cond.sum()

    print("Number of peaks input: ", total_peaks)
    print(
        "Number of peaks filtered because the input/output is on the edge: ",
        num_filtered,
    )
    print("Number of peaks being used: ", peaks_df.shape[0])

    return peaks_df.reset_index(drop=True)


def _regions_to_hot_encoding(
    regions_bed_filename: str,
    genomic_pyfasta: pyfaidx.Fasta,
    hot_encoding_table: np.ndarray,
):
    """
    Encode the seqeunce associated with each region in regions_bed_filename
    to a hot encoded numpy array with shape (len(sequence), len(alphabet)).
    """
    # Get a region (chrom, start, end) from the regions BED file.
    for region in bed.get_regions_from_bed(regions_bed_filename):
        # Region is in BED format: zero-based half open interval.
        chrom, start, end = region
        sequence = str(genomic_pyfasta[chrom][start:end].seq)
        # Hot encode region.
        sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        yield hot_encoding_table[sequence_bytes]


def get_one_hot_encoding_table(
    alphabet="ACGT",
    neutral_alphabet="N",
    neutral_value=0.0,
    dtype=np.float32,
):
    """
    Get an one hot encoding table to encode a DNA sequence to a numpy array.

    Args:
        alphabet (str, optional): Valid nucleotide characters.
        neutral_alphabet (str, optional): Neutral nucleotide characters.
        neutral_value (float, optional): Value for neutral nucleotides.
        dtype (numpy data type, optional): Data type for the encoding table.

    Returns:
        np.ndarray: One-hot encoding table.
    """

    def str_to_uint8(string):
        """Convert string to byte representation."""
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    # Create a np.uint8 matrix of size (255 x 4):
    #   (= all possible values for one byte charater
    #      x one column for each nucleotide in the alphabet)
    # and initialize all of them to zero.
    one_hot_encoding_table = np.zeros(
        (np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype
    )

    # For each ASCII value of the nucleotides used in the alphabet
    # (upper and lower case), set 1 in the correct column.
    one_hot_encoding_table[str_to_uint8(alphabet.upper())] = np.eye(
        len(alphabet), dtype=dtype
    )
    one_hot_encoding_table[str_to_uint8(alphabet.lower())] = np.eye(
        len(alphabet), dtype=dtype
    )

    # For each ASCII value of the nucleotides used in the neutral alphabet
    # (upper and lower case), set neutral_value in the correct column.
    one_hot_encoding_table[str_to_uint8(neutral_alphabet.upper())] = neutral_value
    one_hot_encoding_table[str_to_uint8(neutral_alphabet.lower())] = neutral_value

    return one_hot_encoding_table


def load_data(
    peaks_path,
    bigwigs_dir_path,
    ref_fasta_path,
    output_path,
    input_length,
    target_length,
    num_tasks,
):
    """
    Load peaks, bigwigs, and reference genome for training.

    Args:
        peaks_path (str): Path to the peaks file.
        bigwigs_dir_path (str): Path to the directory containing bigWig files.
        ref_fasta_path (str): Path to the reference genome FASTA file.
        output_path (str): Path to the output directory.
        input_length (int): Length of the input sequence.
        target_length (int): Length of the target sequence.
        num_tasks (int): Number of tasks.

    Returns:
        tuple: DataFrame containing peak regions, labels, and genomic Pyfasta object.
    """
    # Read genomic FASTA file with pyfasta as a memory mapped numpy arrays per chromosome.
    genome_pyfasta = get_genomic_pyfasta(ref_fasta_path, as_string=False)

    # Load bigwigs
    bigwigs_paths = sorted(glob.glob(f"{bigwigs_dir_path}/*.bw"))
    assert (
        len(bigwigs_paths) == num_tasks
    ), "Number of bigwigs should be equal to number of tasks. Check your config file."

    bw = pyBigWig.open(bigwigs_paths[0])

    # Load peaks, extend sequences, and filter edge regions
    extend_sequence_input = partial(extend_sequence, extend_length=input_length)
    peaks_df = pd.read_csv(
        peaks_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    peaks_df[["start_ext", "end_ext"]] = list(
        map(extend_sequence_input, peaks_df["start"], peaks_df["end"])
    )
    peaks_df = filter_edge_regions(peaks_df, bw)

    # Write extended bed file
    output_path = Path(output_path)
    extended_bed_filename = Path("extended_peaks.bed")
    extended_bed_path = output_path.joinpath(extended_bed_filename)
    peaks_df.to_csv(extended_bed_path, sep="\t", index=False, header=False)

    # Write target bed file
    target_bed_filename = Path("target_peaks.bed")
    target_bed_path = output_path.joinpath(target_bed_filename)
    extend_sequence_target = partial(extend_sequence, extend_length=target_length)

    targets_df = pd.DataFrame()
    targets_df["chrom"] = peaks_df["chrom"]
    targets_df[["start", "end"]] = list(
        map(extend_sequence_target, peaks_df["start"], peaks_df["end"])
    )
    targets_df["name"] = (
        targets_df["chrom"]
        + ":"
        + targets_df["start"].astype(str)
        + "-"
        + targets_df["end"].astype(str)
    )
    targets_df.to_csv(target_bed_path, sep="\t", index=False, header=False)

    # Extract bigwig average/max values
    tsv_path = output_path.joinpath(Path("tsv"))
    Path(tsv_path).mkdir(parents=True, exist_ok=True)
    processes = [
        subprocess.Popen(
            "bigWigAverageOverBed {} {} {}.tsv -minMax".format(
                bw_file, target_bed_path, tsv_path.joinpath(Path(bw_file).stem)
            ),
            shell=True,
        )
        for bw_file in bigwigs_paths
    ]
    exitcodes = [p.wait() for p in processes]

    # Create label vectors
    labels_max = np.zeros((peaks_df.shape[0], num_tasks))
    labels_mean = np.zeros((peaks_df.shape[0], num_tasks))
    tsv_files = sorted(glob.glob(f"{tsv_path}/*.tsv"))

    for i, tsv_file in enumerate(tsv_files):
        df = pd.read_csv(tsv_file, sep="\t", header=None)
        df["chrom"] = df[0].str.split(":").str[0]
        df["start"] = df[0].str.split(":").str[1].str.split("-").str[0].astype(int)
        df = df.sort_values(["chrom", "start"])
        labels_max[:, i] = df[7]
        labels_mean[:, i] = df[5]

    # Save labels in numpy
    label_path = output_path.joinpath(Path("labels"))
    Path(label_path).mkdir(parents=True, exist_ok=True)
    np.save(label_path.joinpath(Path("labels_max.npy")), labels_max)
    np.save(label_path.joinpath(Path("labels_mean.npy")), labels_mean)

    # Store them in a dictionary
    labels = {}
    labels["mean"] = labels_mean
    labels["max"] = labels_max

    return peaks_df, labels, genome_pyfasta
