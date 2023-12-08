"""Interpret a trained model using SHAP or ISM."""

import json
import shap
import argparse
import pyfaidx
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
from fastism import FastISM
import tempfile
from tensorflow.keras.models import load_model

import utils.shap as shap_utils


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
        help="Path to BED file",
        default="data/interim/consensus_peaks_2114.bed",
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to trained keras model"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Output path"
    )
    parser.add_argument(
        "-c",
        "--class_index",
        type=maybe_str_or_int,
        default="all",
        help="Index of class to explain",
    )
    parser.add_argument(
        "-me",
        "--method",
        type=str,
        default="shap",
        help="Method to get attribution scores",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=maybe_str_or_int,
        default="all",
        help="Number of regions to interpret",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1024, help="Batch size for ISM"
    )
    parser.add_argument(
        "-correct", "--grad_correct", action="store_true", help="Gradient correction"
    )
    parser.add_argument(
        "-norm", "--normalize", action="store_true", help="Gradient normalization"
    )
    args = parser.parse_args()

    return args


def regions_to_hot_encoding(
    regions_bed_filename: str,
    genomic_pyfasta: pyfaidx.Fasta,
    hot_encoding_table: np.ndarray,
):
    """
    Encode the seqeunce associated with each region in regions_bed_filename
    to a hot encoded numpy array with shape (len(sequence), len(alphabet)).
    """
    # Get a region (chrom, start, end) from the regions BED file.
    for region in _get_region_from_bed(regions_bed_filename):
        # Region is in BED format: zero-based half open interval.
        chrom, start, end = region
        sequence = str(genomic_pyfasta[chrom][start:end].seq)
        # Hot encode region.
        sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        yield hot_encoding_table[sequence_bytes]


def _get_region_from_bed(regions_bed_filename: str):
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


def get_hot_encoding_table(
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
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
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    # 255 x 4
    hot_encoding_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)

    # For each ASCII value of the nucleotides used in the alphabet
    # (upper and lower case), set 1 in the correct column.
    hot_encoding_table[str_to_uint8(alphabet.upper())] = np.eye(
        len(alphabet), dtype=dtype
    )
    hot_encoding_table[str_to_uint8(alphabet.lower())] = np.eye(
        len(alphabet), dtype=dtype
    )

    # For each ASCII value of the nucleotides used in the neutral alphabet
    # (upper and lower case), set neutral_value in the correct column.
    hot_encoding_table[str_to_uint8(neutral_alphabet.upper())] = neutral_value
    hot_encoding_table[str_to_uint8(neutral_alphabet.lower())] = neutral_value

    return hot_encoding_table


def peaks_to_sequences(peaks_bed_file: str, genome_fasta_file: str) -> np.ndarray:
    """Match peaks to genomic sequences and one hot encode the sequences."""
    print("Matching peaks to genomic sequences and one hot encoding the sequences...")
    hot_encoding_table = get_hot_encoding_table()

    with open(peaks_bed_file) as f:
        length_peaks_bed_file = sum(1 for _ in f)

    genomic_pyfasta = pyfaidx.Fasta(genome_fasta_file, sequence_always_upper=True)

    seqs_one_hot = np.zeros((length_peaks_bed_file, 2114, 4))
    for i, hot_encoded_region in tqdm(
        enumerate(
            regions_to_hot_encoding(peaks_bed_file, genomic_pyfasta, hot_encoding_table)
        ),
        total=length_peaks_bed_file,
    ):
        seqs_one_hot[i] = hot_encoded_region
    return seqs_one_hot


def maybe_str_or_int(arg):
    """Convert argument to int or return 'all'."""
    try:
        return int(arg)
    except ValueError:
        pass
    if arg == "all":
        return arg
    raise argparse.ArgumentTypeError("Number must be an int or 'all'")


def region_to_one_hot_encoding(region, genomic_pyfasta, one_hot_encoding_table):
    """
    Convert a genomic region to one-hot encoding.

    Args:
        region (tuple): Genomic region (chromosome, start, end).
        genomic_pyfasta: Pyfasta object for the genome.
        one_hot_encoding_table: One-hot encoding table.

    Returns:
        np.ndarray: One-hot encoded sequence.
    """
    chrom, start, end = region
    # one hot encode region
    return one_hot_encoding_table[
        # Get region sequence as numpy array and convert to uint8.
        genomic_pyfasta[chrom][start:end].view(np.uint8)
    ]


# def get_sequences(regions_df, genomic_pyfasta, one_hot_encoding_table):
#     """
#     Fetch sequences from a given genome.

#     Args:
#         regions_df (pd.DataFrame): DataFrame containing genomic regions.
#         genomic_pyfasta (pyfasta.Fasta): Pyfasta object of the reference genome.
#         one_hot_encoding_table (np.ndarray): One-hot-encoding table.

#     Returns:
#         np.ndarray, np.ndarray: Sequences and a boolean array indicating whether
#         each region was used.
#     """
#     seqs = []
#     regions_used = []

#     for _, row in regions_df.iterrows():
#         region = row["chrom"], row["start_ext"], row["end_ext"]
#         try:
#             seq_ohe = region_to_one_hot_encoding(
#                 region, genomic_pyfasta, one_hot_encoding_table
#             )
#             seqs.append(seq_ohe)
#             regions_used.append(True)
#         except ValueError:
#             print("Invalid bounds! Skipping...")
#             regions_used.append(False)

#     return np.array(seqs), np.array(regions_used)


def interpret(
    model: tf.keras.Model,
    sequences: np.ndarray,
    output_path: str,
    class_index: str = "all",
    method: str = "shap",
    grad_correct: bool = True,
    normalize: bool = True,
    batch_size: int = 1024,
):
    """
    Interpret model predictions using the specified method.

    Args:
        model (tf.keras.Model): Trained model.
        sequences (np.ndarray): Input sequences.
        output_path (str): Output path for saving results.
        class_index (str or int, optional): Index of class to explain.
        method (str, optional): Interpretation method ('shap' or 'ism').
        grad_correct (bool, optional): Correct gradients.
        normalize (bool, optional): Normalize gradients.
        batch_size (int, optional): Batch size for ISM.

    Raises:
        NameError: If the provided method is not recognized.
    """
    print(f"Sequences dimension : {sequences.shape}")

    if method.lower() == "shap":
        # Get the contribution scores from the correct layer
        if model.layers[-1].activation is tf.keras.activations.sigmoid:
            output = model.layers[-2].output
        elif model.layers[-1].activation is tf.keras.activations.softmax:
            output = shap_utils.get_weighted_meannormed_logits(model)
        else:
            output = model.layers[-1].output

        # Get all the contributions or for a specific class
        if class_index == "all":
            output = tf.reduce_sum(output, axis=-1)
        else:
            output = output[:, class_index]

        model_explainer = shap.explainers.deep.TFDeepExplainer(
            (model.input, output),
            shap_utils.shuffle_several_times,
            combine_mult_and_diffref=shap_utils.combine_mult_and_diffref,
        )

        print("Generating shap scores")
        scores = model_explainer.shap_values(sequences, progress_message=100)

        # Normalize and correct gradients
        if grad_correct:
            print("Performing gradient correction")
            scores = scores - np.mean(scores, axis=2, keepdims=True)
        if normalize:
            print("Normalizing gradients")
            scores = scores / np.sqrt(
                np.sum(
                    np.sum(np.square(scores), axis=-1, keepdims=True),
                    axis=-2,
                    keepdims=True,
                )
            )

    elif method.lower() == "ism":
        fast_ism_model = FastISM(model)

        mutations = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        print("Generating ISM scores")
        scores = np.zeros(shape=sequences.shape)
        for idx in range(0, int(np.ceil(sequences.shape[0] / batch_size))):
            # seq_batch has dimensions batch_size x length x 4
            seqs = sequences[idx * batch_size : (idx + 1) * batch_size]
            fast_ism_out = [fast_ism_model(seqs, replace_with=mut) for mut in mutations]

            # transform into batch_size x length  x 4 x num_outputs
            fast_ism_out = np.transpose(np.array(fast_ism_out), (1, 2, 0, 3))

            # Get all the contributions or for a specific class
            if class_index == "all":
                fast_ism_out = tf.reduce_sum(fast_ism_out, axis=-1)
            else:
                fast_ism_out = fast_ism_out[:, :, :, class_index]

            scores[idx * batch_size : (idx + 1) * batch_size,] = fast_ism_out

    else:
        raise NameError("Method should be either 'shap' or 'ism'.")

    assert sequences.shape == scores.shape
    assert sequences.shape[2] == 4

    # TF-MODISCO expects seqs/attributions to be in length-last format
    # (examples, 4, length)
    sequences = np.transpose(sequences, (0, 2, 1)).astype(np.int8)
    scores = np.transpose(scores, (0, 2, 1)).astype(np.float16)

    # Save sequences and scores
    print("Saving 'contribution' scores")
    output_path = Path(output_path)
    sequences_filename = Path("ohe.npz")
    scores_filename = Path("shap.npz")

    np.savez(output_path.joinpath(sequences_filename), sequences)
    np.savez(output_path.joinpath(scores_filename), scores)

    del scores, sequences


def main(args):
    """
    Main function for interpreting model predictions.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    # write all the command line arguments to a json file
    with open(args.output_path + "/interpret_params.json", "w") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # Create a temporary copy of the BED file with only the first 3 columns
    # and max_length regions

    if isinstance(args.number, int):
        num = args.number
    else:
        num = 1000

    with open(args.regions_bed_file, "r") as f:
        lines = f.readlines()
        lines = [line.split("\n") for line in lines]
        lines = lines[:num]
        with tempfile.NamedTemporaryFile(
            mode="w", dir=args.output_path, delete=True
        ) as tmp:
            tmp.writelines(["\n".join(line) for line in lines])
            tmp.flush()

            args.regions_bed_file = tmp.name

            # Convert to one hot encoding
            seqs = peaks_to_sequences(args.regions_bed_file, args.genome_fasta_file)

            # Delete temporary file
            tmp.close()

    model = load_model(args.model, compile=False)
    print("Model loaded successfully")
    model.summary()

    # infer input length
    input_length = model.input_shape[1]
    print("Inferred model input length:", input_length)

    # load genome and one hot encoding table
    # genomic_pyfasta = pyfaidx.Fasta(args.genome_fasta_file, sequence_always_upper=True)
    # one_hot_encoding_table = data.get_one_hot_encoding_table()

    # # load extended sequences
    # extend_sequence_input = partial(data.extend_sequence, extend_length=input_length)
    # regions_df[["start_ext", "end_ext"]] = list(
    #     map(extend_sequence_input, regions_df["start"], regions_df["end"])
    # )

    # seqs, regions_used = get_sequences(
    #     regions_df, genomic_pyfasta, one_hot_encoding_table
    # )
    # save the regions
    # output_path = Path(args.output_path)
    # interpret_bed_filename = Path("interpreted_regions.bed")
    # interpret_bed_path = output_path.joinpath(interpret_bed_filename)
    # regions_df[regions_used].to_csv(
    #     interpret_bed_path, index=False, header=False, sep="\t"
    # )

    interpret(
        model,
        seqs,
        args.output_path,
        args.class_index,
        args.method,
        args.grad_correct,
        args.normalize,
        args.batch_size,
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
