"""Load a trained model and predict on the test set."""
import os
import argparse
import yaml
import tensorflow as tf


def parse_arguments() -> argparse.Namespace:
    """Parse required command line arguments."""
    parser = argparse.ArgumentParser(description="Predict on a model.")
    parser.add_argument(
        "-g",
        "--genome_fasta_file",
        type=str,
        help="Path to genome FASTA file",
        default="data/raw/genome.fa",
    )
    parser.add_argument(
        "-b",
        "--bed_file",
        type=str,
        help="Path to BED file",
        default="data/interim/consensus_peaks_2114.bed",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        type=str,
        help="Path to targets file",
        default="data/interim/targets.npy",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        help="Path to model directory",
        default="checkpoints/",
    )

    return parser.parse_args()
