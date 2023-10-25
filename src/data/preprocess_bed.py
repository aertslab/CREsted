"""
Functions for preprocessing raw ATAC-seq as well as saving matched genome
as input data for the model
"""

import os
import click
import logging
from helpers import bed


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

    chrom_sizes_file = os.path.join(input_folder, "chrom.sizes")
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


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
