"""Interpret a trained model using gradientxinput explainers."""

import argparse
import numpy as np
from datetime import datetime
import os
import pyfaidx
from utils.explain import Explainer, grad_times_input_to_df, plot_attribution_map
from utils.one_hot_encoding import get_hot_encoding_table, regions_to_hot_encoding
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf


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
        "--model_dir",
        type=str,
        help="Path to output model directory (e.g. 'checkpoints/mouse/2023-12-20_15:32/')",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Trained .keras model name inside model_dir/checkpoints. \
        If empty, will load last model in directory.",
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
    output_file: str,
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
        enumerate(class_indices),
        desc="Calculating scores per class for all regions",
        total=n_classes,
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

    np.savez(output_file, scores)
    return scores


def visualize_scores(
    scores: np.ndarray,
    seqs_one_hot: np.ndarray,
    output_dir: str,
    chr_start_ends: str,
    class_indices: list,
    zoom_n_bases: int,
    class_names: list,
    ylim: tuple = None,
    verbose: bool=True,
    savefig: bool=False
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
    if(verbose):
        print(f"Plotting scores and saving to {output_dir}...")
    for seq in tqdm(range(seqs_one_hot.shape[0]), desc="Plotting scores per seq"):
        fig_height_per_class = 2
        fig = plt.figure(figsize=(50, fig_height_per_class * len(class_indices)))
        for i, class_index in enumerate(class_indices):
            seq_class_scores = scores[seq, i, :, :]
            seq_class_x = seqs_one_hot[seq, :, :]
            intgrad_df = grad_times_input_to_df(seq_class_x, seq_class_scores)
            ax = plt.subplot(len(class_indices), 1, i + 1)
            #plt.ylabel(class_names[class_index])
            plot_attribution_map(intgrad_df, ax=ax)
            x_pos = 5  # Adjust this value to set the x-coordinate where you want the text
            y_pos = 0.75*global_max
            text_to_add = class_names[class_index]  
            ax.text(x_pos, y_pos, text_to_add, fontsize=16, ha='left', va='center')

            ax.set_ylim([global_min, global_max])
        plt.xlabel("Position")
        plt.xticks(np.arange(0, zoom_n_bases, 50))
        chr, start, end = chr_start_ends[seq]
        if(ylim is not None):
            plt.ylim(ylim[0],ylim[1])
        if(savefig):
            plt.savefig(os.path.join(output_dir, f"{chr}_{start}_{end}_explained.jpeg"))
            plt.close(fig)
        else:
            plt.show()
    end = datetime.now()
    if(verbose):
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
    regions_bed_file = os.path.join(args.model_dir, "regions.bed")

    classnames = []
    with open(
        os.path.join(args.model_dir, "cell_type_mapping.tsv"), "r"
    ) as cell_mapping:
        for line in cell_mapping:
            classnames.append(line.strip().split("\t")[1])

    # Read regions and one hot encode
    with open(regions_bed_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split("\n") for line in lines]

        # Subset if required
        if args.chromosome is not None:
            print(f"Filtering on chromosome {args.chromosome}")
            lines = [
                line for line in lines if line[0].split("\t")[0] == args.chromosome
            ]
        if args.region_start is not None:
            print(f"Filtering on region >= {args.region_start}")
            lines = [
                line
                for line in lines
                if int(line[0].split("\t")[1]) >= int(args.region_start)
            ]
        if args.region_end is not None:
            print(f"Filtering on region <= {args.region_end}")
            lines = [
                line
                for line in lines
                if int(line[0].split("\t")[2]) <= int(args.region_end)
            ]
    print("Found", len(lines), "regions to explain")
    chr_start_ends = [line[0].split("\t")[0:3] for line in lines]

    # Save lines back to new_region
    new_bed_file = os.path.join(args.model_dir, "regions.explainer.bed")
    with open(new_bed_file, "w") as f:
        for line in lines:
            f.write("\t".join(line) + "\n")

    # One hot encode
    seqs_one_hot = regions_to_hot_encoding(
        regions_bed_filename=new_bed_file,
        genomic_pyfasta=genomic_pyfasta,
        hot_encoding_table=one_hot_encoding_table,
    )

    # Load model
    if args.model_name:
        model = tf.keras.models.load_model(
            os.path.join(args.model_dir, "checkpoints", args.model_name), compile=False
        )
    else:
        model_checkpoints = os.listdir(os.path.join(args.model_dir, "checkpoints"))
        model_checkpoints.sort()
        last_model = model_checkpoints[-1]
        print(f"No model selected. Loading last model {last_model} in directory...")
        model = tf.keras.models.load_model(
            os.path.join(args.model_dir, "checkpoints", last_model), compile=False
        )
    print("Model loaded successfully")

    # Set class index
    if args.class_index == "all":
        args.class_index = [i for i in range(model.output_shape[-1])]
    elif isinstance(eval(args.class_index), list):
        args.class_index = eval(args.class_index)
    elif args.class_index == "combined":
        args.class_index = None

    output_folder = os.path.join(args.model_dir, "evaluation_outputs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    scores = calculate_gradient_scores(
        model,
        seqs_one_hot,
        output_folder,
        class_indices=args.class_index,
        method=args.method,
    )

    output_path = os.path.join(output_folder, f"{args.method}_scores.npy")
    print(f"Saving {args.method} scores to {output_path}...")
    np.save(output_path, scores)

    if args.visualize:
        visualize_scores(
            scores,
            seqs_one_hot,
            output_folder,
            chr_start_ends,
            class_indices=args.class_index,
            zoom_n_bases=args.zoom_n_bases,
            class_names=classnames,
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
