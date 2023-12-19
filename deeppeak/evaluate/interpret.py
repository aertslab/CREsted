"""Interpret a trained model using SHAP or ISM."""

import argparse
import numpy as np
import tempfile
from datetime import datetime
import os
import pyfaidx
from utils.explain import Explainer, grad_times_input_to_df, plot_attribution_map
from utils.one_hot_encoding import get_hot_encoding_table, regions_to_hot_encoding
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model

CLASS_NAMES = os.listdir("data/interim/bw/")
CLASS_NAMES = [name.split(".")[0] for name in CLASS_NAMES]
CLASS_NAMES.sort()


def parse_arguments():
    """Parse command line arguments for interpretations."""
    parser = argparse.ArgumentParser(
        description="Get sequence contribution scores for the model"
    )
    parser.add_argument(
        "-g",
        "--genome_fasta_file",
        type=str,
        help="Path to genome FASTA file",
        default="data/raw/genome.fa",
    )
    parser.add_argument(
        "-r",
        "--regions_bed_file",
        type=str,
        help="Path to BED file which was input to the model",
        default="data/processed/consensus_peaks_inputs.bed",
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to trained keras model"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output dir",
        default="data/output/",
    )
    parser.add_argument(
        "-chr",
        "--chromosome",
        type=str,
        default=None,
        help="Chromosome name to explain. If None, all chromosomes will be explained.",
    )
    parser.add_argument(
        "-rs",
        "--region_start",
        type=int,
        default=None,
        help="Start of regions to explain. If None, all regions will be explained.",
    )
    parser.add_argument(
        "-re",
        "--region_end",
        type=int,
        default=None,
        help="End of regions to explain. If None, all regions will be explained.",
    )
    parser.add_argument(
        "-znb",
        "--zoom_n_bases",
        type=int,
        default=None,
        help="Number of bases to zoom in on in center of region. \
        If None, no zooming will be done.",
    )
    parser.add_argument(
        "-c",
        "--class_index",
        type=str,
        default="all",
        help="Classes to explain. Either 'all', 'combined', or a list of integers. \
        If 'all', all classes will be explained separately. If 'combined', the inputs \
        will be explained respective to the combined model output. If a list of \
        integers, only the classes with the corresponding indices will be explained.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="integrated_grad",
        choices=[
            "integrated_grad",
            "smooth_grad",
            "mutagenesis",
            "saliency",
            "expected_integrated_grad",
        ],
        help="Method to use for interpretation.",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Save plots of interpretation scores.",
    )
    args = parser.parse_args()

    # Ensure that "class_index" is either "all", "best_only", or a list of integers
    assert args.class_index in ["all", "best_only"] or isinstance(
        eval(args.class_index), list
    ), "class_index must be either 'all', 'best_only', or a list of integers"

    return args


def calculate_gradient_scores(
    model: tf.keras.Model,
    x: np.ndarray,
    output_dir: str,
    class_indices: list,
    method: str = "integrated_grad",
):
    if class_indices is not None:
        n_classes = len(class_indices)
    else:
        n_classes = 1  # 'combined' class
        class_indices = [None]
    scores = np.zeros((x.shape[0], n_classes, x.shape[1], x.shape[2]))  # (N, C, W, 4)
    for i, class_index in tqdm(
        enumerate(class_indices), desc="Calculating scores per class", total=n_classes
    ):
        explainer = Explainer(model, class_index=class_index)
        if method == "integrated_grad":
            scores[:, i, :, :] = explainer.integrated_grad(x, baseline_type="zeros")
        elif method == "smooth_grad":
            scores[:, i, :, :] = explainer.smoothgrad(
                x, num_samples=50, mean=0.0, stddev=0.1
            )
        elif method == "mutagenesis":
            scores[:, i, :, :] = explainer.mutagenesis(x, class_index=class_index)
        elif method == "saliency":
            scores[:, i, :, :] = explainer.saliency_maps(x)
        elif method == "expected_integrated_grad":
            scores[:, i, :, :] = explainer.expected_integrated_grad(x, num_baseline=25)
    return scores


def visualize_scores(
    scores: np.ndarray,
    seqs_one_hot: np.ndarray,
    output_dir: str,
    class_indices: list,
    zoom_n_bases: int,
):
    """Visualize interpretation scores."""
    # Center and zoom
    center = int(scores.shape[2] / 2)
    start_idx = center - int(zoom_n_bases / 2)
    scores = scores[:, :, start_idx : start_idx + zoom_n_bases, :]
    seqs_one_hot = seqs_one_hot[:, start_idx : start_idx + zoom_n_bases, :]
    zoom_n_bases = scores.shape[2]

    global_min = scores.min()
    global_max = scores.max()

    # Plot
    now = datetime.now()
    output_dir = os.path.join(output_dir, "interpretation_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Plotting scores and saving to {output_dir}...")
    for seq in tqdm(range(seqs_one_hot.shape[0]), desc="Plotting scores per seq"):
        fig_height_per_class = 2
        fig = plt.figure(figsize=(50, fig_height_per_class * len(class_indices)))
        for i, class_index in enumerate(class_indices):
            seq_class_scores = scores[seq, i, :, :]
            seq_class_x = seqs_one_hot[seq, :, :]
            intgrad_df = grad_times_input_to_df(seq_class_x, seq_class_scores)
            ax = plt.subplot(len(class_indices), 1, i + 1)
            plt.ylabel(CLASS_NAMES[class_index])
            plot_attribution_map(intgrad_df, ax=ax)

            ax.set_ylim([global_min, global_max])
        plt.xlabel("Position")
        plt.xticks(np.arange(0, zoom_n_bases, 50))
        plt.savefig(os.path.join(output_dir, f"seq_{seq}.jpeg"))
        plt.close(fig)
    end = datetime.now()
    print(f"Plotting took {end - now}")


def main(args):
    """
    Main function for interpreting model predictions.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    # Load data
    one_hot_encoding_table = get_hot_encoding_table()
    genomic_pyfasta = pyfaidx.Fasta(args.genome_fasta_file, sequence_always_upper=True)

    # Read regions and one hot encode
    with open(args.regions_bed_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split("\n") for line in lines]

        # Subset if required
        if args.chromosome is not None:
            lines = [
                line for line in lines if line[0].split("\t")[0] == args.chromosome
            ]
        if args.region_start is not None:
            lines = [
                line
                for line in lines
                if int(line[0].split("\t")[1]) >= args.region_start
            ]
        if args.region_end is not None:
            lines = [
                line for line in lines if int(line[0].split("\t")[2]) <= args.region_end
            ]

        # One hot encode
        width_region = int(lines[0][0].split("\t")[2]) - int(lines[0][0].split("\t")[1])
        seqs_one_hot = np.zeros((len(lines), width_region, 4))

        print(f"One hot encoding {len(lines)} regions...")
        with tempfile.NamedTemporaryFile(
            mode="w", dir=args.output_dir, delete=True
        ) as tmp:
            tmp.writelines(["\n".join(line) for line in lines])
            tmp.flush()

            args.regions_bed_file = tmp.name

            # Convert to one hot encoding
            for i, hot_encoded_region in tqdm(
                enumerate(
                    regions_to_hot_encoding(
                        args.regions_bed_file, genomic_pyfasta, one_hot_encoding_table
                    )
                ),
                total=len(lines),
            ):
                seqs_one_hot[i] = hot_encoded_region

            # Delete temporary file
            tmp.close()

    model = load_model(args.model, compile=False)
    model.summary()
    print("Model loaded successfully")

    # Set class index
    if args.class_index == "all":
        args.class_index = [i for i in range(model.output_shape[-1])]
    elif isinstance(eval(args.class_index), list):
        args.class_index = eval(args.class_index)
    elif args.class_index == "combined":
        args.class_index = None

    scores = calculate_gradient_scores(
        model,
        seqs_one_hot,
        args.output_dir,
        class_indices=args.class_index,
        method=args.method,
    )

    output_path = os.path.join(args.output_dir, f"{args.method}_scores.npy")
    print(f"Saving {args.method} scores to {output_path}...")
    np.save(output_path, scores)

    if args.visualize:
        visualize_scores(
            scores,
            seqs_one_hot,
            args.output_dir,
            class_indices=args.class_index,
            zoom_n_bases=args.zoom_n_bases,
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
