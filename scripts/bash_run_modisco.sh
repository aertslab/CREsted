#!/bin/bash
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -p hmem_128C_256T_2TB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niklas.kempynck@kuleuven.be

# Define the folder path containing the .npz files
folder_path="/data/projects/c04/cbd-saerts/nkemp/mouse/wmb/top_500/contribution_scores/"

source /data/projects/c04/cbd-saerts/nkemp/software/anaconda3/etc/profile.d/conda.sh
conda activate deeppeak

# Loop through the folder to find unique cell types
for file in "${folder_path}"/*_shaps.npz; do
    # Extract cell type from filename
    cell_type=$(basename "$file" "_shaps.npz")

    # Define filenames for ohe and shap files, and output files
    ohe_file="${folder_path}/${cell_type}_ohs.npz"
    shap_file="${folder_path}/${cell_type}_shaps.npz"
    output_file="${folder_path}/${cell_type}_modisco_results.h5"
    report_dir="${folder_path}/${cell_type}_report/"

    # Check if the modisco results .h5 file does not exist for the cell type
    if [[ ! -f "$output_file" ]]; then
        if [[ -f "$ohe_file" ]] && [[ -f "$shap_file" ]]; then
            # Run the modisco motifs command
            modisco motifs -s "$ohe_file" -a "$shap_file" -n 2000 -o "$output_file"
        else
            echo "Missing files for cell type: $cell_type"
        fi
 # Check if the modisco results .h5 file was successfully created
        #if [[ -f "$output_file" ]] && ; then
            # Run the modisco report command
            #modisco report -i "$output_file" -o "$report_dir" -s "$report_dir" -m /data/projects/c04/cbd-saerts/nkemp/tools/motifs.meme
        #else
            #echo "Modisco motifs command failed for cell type: $cell_type"
        #fi
    else
        echo "Modisco results already exist for cell type: $cell_type"
    fi
done
