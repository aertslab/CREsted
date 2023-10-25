import os
import click
import numpy as np
import pyfaidx
from tqdm import tqdm
from helpers import genome, bed


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
    print("Matching peaks to genomic sequences and one hot encoding the sequences...")
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
    return seqs_one_hot


@click.command()
@click.argument("input_folder", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path(exists=True))
def main(input_folder, output_folder):
    print("\nCreating input data...")
    peaks_bed_name = "consensus_peaks"
    genome_fasta_file = os.path.join(input_folder, "genome.fa")

    # Create input data (consensus peak 2114 sequences)
    seqs_one_hot = peaks_to_sequences(
        os.path.join(output_folder, f"{peaks_bed_name}_2114.bed"), genome_fasta_file
    )

    print(f"Saving input data (seqs_one_hot) to {output_folder}...")
    np.save(os.path.join(output_folder, "peaks_one_hot.npy"), seqs_one_hot)


if __name__ == "__main__":
    main()
