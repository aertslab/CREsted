##TRAINING MODEL
workdir=/staging/leuven/stg_00002/lcb/lcb_projects/biccn_okt_2023/analysis/deeppeak/foundation/data/2114/cbpnet_custom/

source /staging/leuven/stg_00002/lcb/nkemp/software/anaconda3/etc/profile.d/conda.sh
conda activate deeplearning_py38_tf241

python /data/leuven/347/vsc34783/DeepTopic+/main.py \
    --runType train \
    --inputtopics tmp \
    --numTopics 19 \
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
    --transferlearn False \
    --insertmotif False \
    --wandbname biccn_foundation_deeppeak \
    --wandbUser kemp \
    --gpuname 0 \
    --useclassweight False \
    --epochs 200 \
    --patience 5 \
    --batchsize 256 \
    --stride 50 \
    --useCreatedData /staging/leuven/stg_00002/lcb/lcb_projects/biccn_okt_2023/analysis/deeppeak/foundation/data/2114/ \
    --doubleDATArc True \
    --outputdirc $workdir \
    --namehdf5 False \
    --model chrombpnet \
    --selectedtopicsShap False
