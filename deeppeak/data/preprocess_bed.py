"""
Functions for preprocessing raw ATAC-seq as well as saving matched genome
as input data for the model
"""

import os
import yaml
import argparse
from helpers import bed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw ATAC-seq data and create input data for the model."
    )
    parser.add_argument(
        "-r",
        "--regions_bed_file",
        type=str,
        default="data/raw/consensus_peaks.bed",
        help="Path to the input regions bed file.",
    )
    parser.add_argument(
        "-c",
        "--chrom_sizes_file",
        type=str,
        default="data/raw/chrom.sizes",
        help="Path to the chromosome sizes file. Required if --filter_chrom is True.",
    )
    parser.add_argument(
        "-it",
        "--inputs_or_targets",
        type=str,
        required=True,
        help="Which bed output file to create. Either 'inputs' or 'targets'.",
    )
    parser.add_argument("--filter_negative", type=bool, default=True)
    parser.add_argument("--filter_chrom", type=bool, default=True)
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="data/processed/",
        help="Path to the folder where the processed data will be saved.",
    )
    # check if chrom_sizes_file exists if filter_chrom is True
    args = parser.parse_args()
    if args.filter_chrom and not os.path.exists(args.chrom_sizes_file):
        raise ValueError(
            f"Chromosome sizes file not found at {args.chrom_sizes_file}. "
            "Please specify the correct path to the chromosome sizes file or set \
            --filter_chrom to False."
        )

    return args


# BED file processing
def preprocess_bed(
    input_path: str,
    output_path: str,
    chrom_sizes_file: str,
    n_extend: int,
    filter_negative: bool = False,
    filter_chrom: bool = False,
):
    """
    Preprocess a BED file by extending the start and end positions by a given
    value, and filtering out negative and out of bounds coordinates.
    """
    print(f"\nPreprocessing BED file: {input_path} to {output_path}...")
    if n_extend > 0:
        print(f"Extending start and end positions by {n_extend}...")
        bed.extend_bed_file(input_path, output_path, n_extend)

    if filter_negative:
        print("Filtering out negative coordinates...")
        bed.filter_bed_negative_regions(output_path, output_path)

    if filter_chrom:
        print("Filtering out out of bounds coordinates...")
        bed.filter_bed_chrom_regions(output_path, output_path, chrom_sizes_file)
    print("Done!")


def main(args: argparse.Namespace, config: dict):
    """Run bed preprocessing pipeline."""
    if args.inputs_or_targets == "inputs":
        final_regions_width = int(config["seq_len"])
    elif args.inputs_or_targets == "targets":
        final_regions_width = int(config["target_len"])
    else:
        raise ValueError(
            "Please specify either 'inputs' or 'targets' for --inputs_or_targets."
        )
    regions_bed_name = os.path.basename(args.regions_bed_file).split(".")[0]
    regions_width = bed.get_bed_region_width(args.regions_bed_file)
    n_extend = (final_regions_width - regions_width) // 2

    # Preprocess peaks BED file
    preprocess_bed(
        input_path=args.regions_bed_file,
        output_path=os.path.join(
            args.output_folder, f"{regions_bed_name}_{args.inputs_or_targets}.bed"
        ),
        chrom_sizes_file=args.chrom_sizes_file,
        n_extend=n_extend,
        filter_negative=args.filter_negative,
        filter_chrom=args.filter_chrom,
    )


if __name__ == "__main__":
    with open("configs/user.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_arguments()
    main(args, config)
