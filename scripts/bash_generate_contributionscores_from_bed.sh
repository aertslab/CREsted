#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH -p gpu_h100_64C_128T_2TB
#SBATCH -G 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niklas.kempynck@kuleuven.be

# Define parameters

INPUT_BED_DIR="/data/projects/c04/cbd-saerts/nkemp/mouse/wmb/top_500/bed/"
MODEL_LOCATION="/data/projects/c04/cbd-saerts/nkemp/mouse/wmb/deeppeak/runs/TL_bigboy/checkpoints/17.keras"
MAPPING_FILE='/data/projects/c04/cbd-saerts/nkemp/mouse/wmb/top_500/mapping.txt'
OUTPUT_DIRECTORY="/data/projects/c04/cbd-saerts/nkemp/mouse/wmb/top_500/contribution_scores/"
GENOME="/data/projects/c04/cbd-saerts/nkemp/software/dev_DeepPeak/DeepPeak/data/raw/genome.fa"
LOG_FILE="logfile_cbs_calcs_seqsonly.log"

# Write parameter information to the log file
{
  echo "Input BED Directory: $INPUT_BED_DIR"
  echo "Model Location: $MODEL_LOCATION"
  echo "Mapping File:" $MAPPING_FILE
  echo "Output Directory: $OUTPUT_DIRECTORY"
  echo "Genome: $GENOME"
  echo "Log File: $LOG_FILE"
  echo "--------------------------------"
} > $LOG_FILE

echo "GPU status before running the script:"
nvidia-smi

source /data/projects/c04/cbd-saerts/nkemp/software/anaconda3/etc/profile.d/conda.sh
conda activate deeppeak
ml load CUDA/11.8.0
#module load NCCL/2.16.2-GCCcore-11.3.0-CUDA-11.8.0

python generate_contributionscores_from_bed.py --input_dir $INPUT_BED_DIR --model_path $MODEL_LOCATION --mapping $MAPPING_FILE --genome $GENOME --output_dir $OUTPUT_DIRECTORY >> $LOG_FILE 2>&1
