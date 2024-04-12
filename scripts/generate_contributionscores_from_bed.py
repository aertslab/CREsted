import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import os
import pyfaidx

import sys
sys.path.append('../deeppeak/evaluate')
from utils.explain import Explainer, grad_times_input_to_df, plot_attribution_map
from utils.one_hot_encoding import regions_to_hot_encoding, get_hot_encoding_table

from interpret import *
def make_mapping_dict(mapping_file):
    dictionary = {}
    with open(mapping_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')  # Split each line on tab
            if len(parts) == 2:
                cell_type, index = parts
                dictionary[cell_type] = int(index)  # Assuming index is an integer
    return dictionary

def load_your_model(model_path):
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

def main(input_dir, model_path, mapping, genome, output_dir):
    print(tf.config.list_physical_devices('GPU'))
    model = load_your_model(model_path)
    mapping_dict = make_mapping_dict(mapping)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Made new output directory at '+output_dir+'.')

    try:
        genomic_pyfasta = pyfaidx.Fasta(genome, sequence_always_upper=True)
    except Exception as e:
        print(f"Error loading genome with pyfaidx: {e}", file=sys.stderr)
        sys.exit(1)
    hot_encoding_table = get_hot_encoding_table()
    for i, bed_file in tqdm(enumerate(os.listdir(input_dir)), total=len(os.listdir(input_dir))):
        if bed_file.endswith(".bed"):
            try:
                cell_type = bed_file.split('.')[0]
                model_class = mapping_dict[cell_type]
                print('Model class: '+str(model_class))
                outfile_shaps = cell_type+'_shaps.npz'
                outfile_seqs = cell_type+'_ohs.npz'
                
                bed_file_path = os.path.join(input_dir, bed_file)
                seqs_onehot = regions_to_hot_encoding(bed_file_path, genomic_pyfasta, hot_encoding_table)
                np.savez(os.path.join(output_dir, outfile_seqs), seqs_onehot)
                
                #contribution_scores = calculate_gradient_scores(
                #        model,
                #        seqs_onehot,
                #        os.path.join(output_dir, outfile_shaps),
                #        class_indices=[model_class],
                #        method='expected_integrated_grad',
                #)
                print('Completed calculations for '+cell_type)
            except Exception as e:
                print(f"Failed processing {bed_file}: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BED files to generate contribution scores.")
    parser.add_argument('--input_dir', required=True, help='Directory containing BED files')
    parser.add_argument('--model_path', required=True, help='Path to the deep learning model')
    parser.add_argument('--mapping', required=True, help='Path to mapping of cell types to model classes')
    parser.add_argument('--output_dir', required=True, help='Directory to save output NPZ files')
    parser.add_argument('--genome', required=True, help='Path to reference genome')

    args = parser.parse_args()

    main(args.input_dir, args.model_path, args.mapping, args.genome, args.output_dir)
