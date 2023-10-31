#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=150G
#SBATCH --time=36:00:00
#SBATCH -A lp_big_wice_gpu
#SBATCH -p dedicated_big_gpu
#SBATCH -G 1
#SBATCH --cluster=wice
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niklas.kempynck@kuleuven.be

##TRAINING MODEL
workdir=/staging/leuven/stg_00002/lcb/lcb_projects/biccn_okt_2023/analysis/deeppeak/human/data/2114/cbpnet_custom_TL/

source /staging/leuven/stg_00002/lcb/nkemp/software/anaconda3/etc/profile.d/conda.sh
conda activate deeplearning_py38_tf241

python /data/leuven/347/vsc34783/DeepTopic+/main.py \
    --runType train \
    --inputtopics tmp \
    --numTopics 20 \
    --selectedtopics False \
    --mergetopic False \
    --consensuspeaks tmp \
    --genome tmp \
    --chrSize tmp \
    --validchroms 10 \
    --testchroms 10 \
    --activation gelu \
    --numkernels 512 \
    --motifwidth 5 \
    --maxpoolsize 2 \
    --convDO 0 \
    --denseDO 0.3 \
    --conv_l2 1e-6 \
    --dense_l2 1e-4 \
    --numdense 1024 \
    --usetransformer False \
    --seqlen 2114 \
    --seed 777 \
    --learningrate 0.001 \
    --transferlearn /staging/leuven/stg_00002/lcb/lcb_projects/biccn_okt_2023/analysis/deeppeak/human/data/2114/cbpnet_custom/model_epoch_21.hdf5 \
    --insertmotif False \
    --wandbname biccn_human_deeppeak \
    --wandbUser kemp \
    --gpuname 0 \
    --useclassweight False \
    --epochs 200 \
    --patience 5 \
    --batchsize 256 \
    --stride 50 \
    --useCreatedData /staging/leuven/stg_00002/lcb/lcb_projects/biccn_okt_2023/analysis/deeppeak/human/data/2114/TL/ \
    --doubleDATArc True \
    --outputdirc $workdir \
    --namehdf5 False \
    --model chrombpnet \
    --selectedtopicsShap False