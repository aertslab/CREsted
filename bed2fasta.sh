
outdirbed=/staging/leuven/stg_00002/lcb/lcb_projects/biccn_okt_2023/analysis/deeppeak/mouse/results/rankings/20231014/ranked_files/top_500/bed/
outdirfasta=/staging/leuven/stg_00002/lcb/lcb_projects/biccn_okt_2023/analysis/deeppeak/mouse/results/rankings/20231014/ranked_files/top_500/fasta/
twobitf=/staging/leuven/res_00001/genomes/mus_musculus/mm10_ucsc/fasta/mm10.2bit

source ~/.bash_profile

module load Kent_tools/20181218-linux.x86_64

### FASTA
if [ ! -d $outdirfasta ]
then
    mkdir -p $outdirfasta
else
    echo "Output dir '$outdirfasta' exists."
fi

for f in $outdirbed/*.bed
do
    outf=${f##*/}
    outf=$outdirfasta/${outf%.bed}.fa

    tempf=`mktemp tmpbedXXXXXXX.bed`

    # account for difference to pyranges (1-based)
    awk -F"\t" '{ OFS="\t"; print $1, ($2-1), ($3-1), $4 }' $f > $tempf
    twoBitToFa -bed=$tempf $twobitf $outf
    rm $tempf

    echo "written '$outf' ..."
done
echo "All FASTA files done."

echo "ALL DONE."
