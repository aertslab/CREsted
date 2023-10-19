#!/bin/bash

# Initialize variables with default values
bed_file=""
chrom_sizes_file=""
temp_file="temp_filtered.bed"

# Parse command-line arguments
while getopts "b:c:t:" opt; do
  case $opt in
    b)
      bed_file="$OPTARG"
      ;;
    c)
      chrom_sizes_file="$OPTARG"
      ;;
    t)
      temp_file="$OPTARG"
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

# Clear temp file if it exists
> "$temp_file"

# Loop through each line of the chromosome sizes file
while IFS=$'\t' read -r chrom size; do
  # Filter the BED file based on chromosome and size, appending to temp file
  awk -v chrom="$chrom" -v size="$size" '$1 == chrom && $3 <= size' "$bed_file" >> "$temp_file"
done < "$chrom_sizes_file"

# Sort the temporary file (optional)
sort -k1,1 -k2,2n "$temp_file" > "$bed_file"

# Remove the temporary file
rm "$temp_file"

echo "Filtering complete. The filtered BED file is saved."

