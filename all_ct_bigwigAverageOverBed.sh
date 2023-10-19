#!/bin/bash

# Initialize variables with default values
output_dir=""
bigwig_dir=""
peak_bed_file=""

# Parse command-line arguments
while getopts "o:b:p:" opt; do
  case $opt in
    o)
      output_dir="$OPTARG"
      ;;
    b)
      bigwig_dir="$OPTARG" # IMPORTANT TO ONLY LOOK AT 1000 bp PEAKS AND NOT 2114 ! 
      ;;
    p)
      peak_bed_file="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Create the output directory if it does not exist
mkdir -p $output_dir

module load Kent_tools

# Run bigWigAverageOverBed in parallel for all .bw files in the directory
ls -1 $bigwig_dir/*.bw | parallel -j 8 "start=\$(date +%s.%N); bigWigAverageOverBed -minMax {} $peak_bed_file $output_dir/{/}.tsv; end=\$(date +%s.%N); echo \$(echo \"\$end - \$start\" | bc) seconds >> $output_dir/time_log.txt"





