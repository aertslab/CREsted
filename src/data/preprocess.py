"""Functions for preprocessing raw ATAC-seq and genomic data."""

import os
import tempfile
from contextlib import contextmanager

######### General functions #########
def _raw_assertion(path: str):
    """Assert that a file is not in the "raw" directory."""
    assert 'raw' not in path, f"Out file {path} is in the raw directory. Select a different directory."

@contextmanager
def smart_open(input_path, output_path):
    """Open a file for reading and another for writing, handling the case where paths are the same."""
    _raw_assertion(output_path) 

    if input_path == output_path:
        dir_name = os.path.dirname(input_path)
        with tempfile.NamedTemporaryFile(mode='w', dir=dir_name, delete=False) as tmpfile:
            yield open(input_path, 'r'), tmpfile
            temp_name = tmpfile.name
        os.replace(temp_name, output_path)
    else:
        with open(output_path, 'w') as outfile:
            yield open(input_path, 'r'), outfile

######### BED file processing #########
def extend_bed_file(input_path: str, output_path: str, value: int):
    """
    Extend the start and end positions of a BED file by a given value.
    """
    with smart_open(input_path, output_path) as (infile, outfile):
        for line in infile:
            cols = line.strip().split()
            cols[1] = str(int(cols[1]) - value)
            cols[2] = str(int(cols[2]) + value)
            outfile.write('\t'.join(cols) + '\n')

def filter_bed_negative_regions(input_path: str, output_path: str):
    """
    Filters out lines from a BED file that have negative values in the second or third column.
    """
    with smart_open(input_path, output_path) as (infile, outfile):
        for line_number, line in enumerate(infile, start=1):
            cols = line.strip().split()
            if int(cols[1]) < 0 or int(cols[2]) < 0:
                print(f"Negative coordinate found on line: {line_number}")
                continue
            outfile.write(line)

def filter_bed_chrom_regions(input_path: str, output_path: str, chrom_sizes_file: str):
    """
    Filters out lines from a BED file that are out of bounds of the chromosome size.
    """
    filtered_lines = []
    total_lines = 0
    chrom_sizes = {}

    with open(chrom_sizes_file, 'r') as sizes:
        for line in sizes:
            chrom, size = line.strip().split('\t')
            size = int(size)
            chrom_sizes[chrom] = size

    with smart_open(input_path, output_path) as (infile, outfile):
        for bed_line in infile:
            total_lines += 1
            bed_cols = bed_line.strip().split('\t')
            chrom = bed_cols[0]
            if chrom in chrom_sizes and int(bed_cols[2]) <= chrom_sizes[chrom]:
                filtered_lines.append(bed_line)

    # Sort the filtered BED file
    sorted_lines = sorted(filtered_lines, key=lambda x: (x.split('\t')[0], int(x.split('\t')[1])))

    with open(output_path, 'w') as outfile:
        outfile.writelines(sorted_lines)

    print(f"chromosome size: Filtered out {total_lines - len(sorted_lines)} lines out of {total_lines} total lines.")

if __name__ == '__main__':
    extend_bed_file('data/raw/consensus_peaks_bicnn.bed', 'data/interim/consensus_peaks_bicnn_2114.bed', 807)
    filter_bed_negative_regions("data/interim/consensus_peaks_bicnn_2114.bed", "data/interim/consensus_peaks_bicnn_2114.bed")
    filter_bed_chrom_regions("data/interim/consensus_peaks_bicnn_2114.bed", "data/interim/consensus_peaks_bicnn_2114.bed", "data/raw/mm10.chrom.sizes")

    extend_bed_file('data/raw/consensus_peaks_bicnn.bed', 'data/interim/consensus_peaks_bicnn_1000.bed', 250)
    filter_bed_negative_regions("data/interim/consensus_peaks_bicnn_1000.bed", "data/interim/consensus_peaks_bicnn_1000.bed")
    filter_bed_chrom_regions("data/interim/consensus_peaks_bicnn_1000.bed", "data/interim/consensus_peaks_bicnn_1000.bed", "data/raw/mm10.chrom.sizes")


