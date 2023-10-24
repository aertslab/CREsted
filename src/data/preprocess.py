"""Functions for preprocessing raw ATAC-seq and genomic data."""

import os
import click
import logging
import tempfile
import numpy as np
import pyfaidx
from pybedtools import BedTool
from contextlib import contextmanager
from helpers.bed import extend_bed_file, filter_bed_negative_regions, filter_bed_chrom_regions


# General functions
def check_data_folder(folder_path: str):
    """Check that all required files are in the data folder.
    """
    required_files = [
        "consensus_peaks.bed",
        "bw/"
        "chrom.sizes",
        "genome.fa"
    ]
    for file in required_files:
        assert os.path.exists(os.path.join(folder_path, file)), f"{file} not found in {folder_path}"


# BED file processing
def bed_preprocessing(
    input_path: str,
    output_path: str,
    chrom_sizes_file: str,
    value: int = 0,
    filter_negative: bool = False,
    filter_chrom: bool = False
):
    """
    Preprocess a BED file by extending the start and end positions by a given
    value, and filtering out negative and out of bounds coordinates.
    """
    print(f"Preprocessing BED file: {input_path} to {output_path}...")
    if value > 0:
        extend_bed_file(input_path, output_path, value)

    if filter_negative:
        filter_bed_negative_regions(input_path, output_path)

    if filter_chrom:
        filter_bed_chrom_regions(input_path, output_path, chrom_sizes_file)


# One hot encoding
def _get_regions_from_bed(regions_bed_filename: str):
    """
    Read BED file and yield a region (chrom, start, end) for each invocation.
    """
    with open(regions_bed_filename, "r") as fh_bed:
        for line in fh_bed:
            line = line.rstrip("\r\n")

            if line.startswith("#"):
                continue

            columns = line.split("\t")
            chrom = columns[0]
            start, end = [int(x) for x in columns[1:3]]
            region = chrom, start, end
            yield region


def regions_to_hot_encoding(
    regions_bed_filename: str,
    genomic_pyfasta: pyfaidx.Fasta,
    hot_encoding_table: np.ndarray
):
    """
    Encode the seqeunce associated with each region in regions_bed_filename
    to a hot encoded numpy array with shape (len(sequence), len(alphabet)).
    """
    # Get a region (chrom, start, end) from the regions BED file.
    for region in _get_regions_from_bed(regions_bed_filename):
        # Region is in BED format: zero-based half open interval.
        chrom, start, end = region
        sequence = str(genomic_pyfasta[chrom][start:end].seq)
        # Hot encode region.
        sequence_bytes = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
        yield hot_encoding_table[sequence_bytes]


def get_hot_encoding_table(
    alphabet: str = 'ACGT',
    neutral_alphabet: str = 'N',
    neutral_value: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    """
    Get hot encoding table to encode a DNA sequence to a numpy array with shape
    (len(sequence), len(alphabet)) using bytes.
    """
    def str_to_uint8(string) -> np.ndarray:
        """
        Convert string to byte representation.
        """
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)

    # 255 x 4
    hot_encoding_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)

    # For each ASCII value of the nucleotides used in the alphabet
    # (upper and lower case), set 1 in the correct column.
    hot_encoding_table[str_to_uint8(alphabet.upper())] = np.eye(len(alphabet), dtype=dtype)
    hot_encoding_table[str_to_uint8(alphabet.lower())] = np.eye(len(alphabet), dtype=dtype)

    # For each ASCII value of the nucleotides used in the neutral alphabet
    # (upper and lower case), set neutral_value in the correct column.
    hot_encoding_table[str_to_uint8(neutral_alphabet.upper())] = neutral_value
    hot_encoding_table[str_to_uint8(neutral_alphabet.lower())] = neutral_value

    return hot_encoding_table


# Main preprocessing pipeline
@click.command()
@click.argument('input_folder', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path(exists=True))
def main(
    input_folder: str,
    output_folder: str,
):
    """Run data preprocessing pipeline to turn raw data into processed data (../processed)"""
    # Check folders and paths
    check_data_folder(input_folder)

    bigwigs_folder = os.path.join(input_folder, "bw")
    chrom_sizes_file = os.path.join(input_folder, "chrom.sizes")
    genome_fasta_file = os.path.join(input_folder, "genome.fa")
    peaks_bed_file = os.path.join(input_folder, "consensus_peaks.bed")

    peaks_bed_name = os.path.basename(peaks_bed_file).split('.')[0]

    # Preprocess peaks BED file
    bed_preprocessing(
        input_path=peaks_bed_file,
        output_path=f'data/interim/{peaks_bed_name}_2114.bed',
        chrom_sizes_file=chrom_sizes_file,
        value=807,
        filter_negative=True,
        filter_chrom=True
    )  # 2114 peaks

    bed_preprocessing(
        input_path=peaks_bed_file,
        output_path=f'data/interim/{peaks_bed_name}_1000.bed',
        chrom_sizes_file=chrom_sizes_file,
        value=250,
        filter_negative=True,
        filter_chrom=True
    )  # 1000 peaks


    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
