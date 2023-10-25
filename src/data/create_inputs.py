"""
Functions for preprocessing raw ATAC-seq as well as saving matched genome
as input data for the model
"""

import os
import click
import logging
import numpy as np
import pyfaidx
from tqdm import tqdm
from pybedtools import BedTool
from contextlib import contextmanager
from helpers import bed, genome


# General functions
def check_data_folder(folder_path: str):
    """Check that all required files are in the data folder."""
    required_files = ["consensus_peaks.bed", "bw/", "chrom.sizes", "genome.fa"]
    for file in required_files:
        assert os.path.exists(
            os.path.join(folder_path, file)
        ), f"{file} not found in {folder_path}"


# BED file processing
def preprocess_bed(
    input_path: str,
    output_path: str,
    chrom_sizes_file: str,
    value: int,
    filter_negative: bool = False,
    filter_chrom: bool = False,
):
    """
    Preprocess a BED file by extending the start and end positions by a given
    value, and filtering out negative and out of bounds coordinates.
    """
    print(f"Preprocessing BED file: {input_path} to {output_path}...")
    if value > 0:
        bed.extend_bed_file(input_path, output_path, value)

    if filter_negative:
        bed.filter_bed_negative_regions(output_path, output_path)

    if filter_chrom:
        bed.filter_bed_chrom_regions(output_path, output_path, chrom_sizes_file)


# Linking regions to genomic data
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


def peaks_to_sequences(peaks_bed_file: str, genome_fasta_file: str) -> np.ndarray:
    """Match peaks to genomic sequences and one hot encode the sequences."""
    print(
        f"\nMatching peaks to genomic sequences and one hot encoding the sequences..."
    )
    hot_encoding_table = genome.get_hot_encoding_table()

    with open(peaks_bed_file) as f:
        length_peaks_bed_file = sum(1 for _ in f)

    genomic_pyfasta = pyfaidx.Fasta(genome_fasta_file, sequence_always_upper=True)

    seqs_one_hot = np.zeros((length_peaks_bed_file, 2114, 4))
    for i, hot_encoded_region in tqdm(
        enumerate(
            _regions_to_hot_encoding(
                peaks_bed_file, genomic_pyfasta, hot_encoding_table
            )
        ),
        total=length_peaks_bed_file,
    ):
        seqs_one_hot[i] = hot_encoded_region


# Main preprocessing pipeline
@click.command()
@click.argument("input_folder", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path(exists=True))
def main(
    input_folder: str,
    output_folder: str,
):
    """Run data preprocessing pipeline to turn raw bed and genome
    sequences into input data for the model.
    """
    # Check folders and paths
    check_data_folder(input_folder)

    bigwigs_folder = os.path.join(input_folder, "bw")
    chrom_sizes_file = os.path.join(input_folder, "chrom.sizes")
    genome_fasta_file = os.path.join(input_folder, "genome.fa")
    peaks_bed_file = os.path.join(input_folder, "consensus_peaks.bed")

    peaks_bed_name = os.path.basename(peaks_bed_file).split(".")[0]

    # Preprocess peaks BED file
    preprocess_bed(
        input_path=peaks_bed_file,
        output_path=os.path.join(output_folder, f"{peaks_bed_name}_2114.bed"),
        chrom_sizes_file=chrom_sizes_file,
        value=807,
        filter_negative=True,
        filter_chrom=True,
    )  # 2114 peaks (for inputs)

    preprocess_bed(
        input_path=peaks_bed_file,
        output_path=os.path.join(output_folder, f"{peaks_bed_name}_1000.bed"),
        chrom_sizes_file=chrom_sizes_file,
        value=250,
        filter_negative=True,
        filter_chrom=True,
    )  # 1000 peaks (for targets)

    # Create input data (consensus peak 2114 sequences)
    seqs_one_hot = peaks_to_sequences(
        os.path.join(output_folder, f"{peaks_bed_name}_2114.bed"), genome_fasta_file
    )

    print(f"\nSaving input data (seqs_one_hot) to {output_folder}...")
    np.save(os.path.join(output_folder, "peaks_one_hot.npy"), seqs_one_hot)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
