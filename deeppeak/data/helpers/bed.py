"""Functions for preprocessing ATAC-seq peak bed files."""

import os
import tempfile
import shutil
import numpy as np
from contextlib import contextmanager
import numpy as np


def _raw_assertion(path: str):
    """Assert that a file is not in the "raw" directory."""
    assert (
        "raw" not in path
    ), f"Out file {path} is in the raw directory. \
    Select a different directory."


def extend_bed_file(input_path: str, output_path: str, final_regions_width: int):
    """
    Extend the start and end positions of a BED file by a given value.
    """
    with open(input_path, "r") as infile:
        lines = infile.readlines()
    with open(output_path, "w") as outfile:
        for line in lines:
            cols = line.strip().split()
            current_width = int(cols[2]) - int(cols[1])
            if current_width == final_regions_width:
                outfile.write(line)
            else:
                width_diff = final_regions_width - current_width
                if width_diff % 2 == 0:
                    n_extend = width_diff // 2
                    cols[1] = str(int(cols[1]) - n_extend)
                    cols[2] = str(int(cols[2]) + n_extend)
                else:
                    n_extend = width_diff // 2
                    cols[1] = str(int(cols[1]) - n_extend)
                    cols[2] = str(int(cols[2]) + n_extend + 1)
                assert (
                    int(cols[2]) - int(cols[1]) == final_regions_width
                ), f"cols: {cols}"
                outfile.write("\t".join(cols) + "\n")


def filter_bed_negative_regions(input_path: str, output_path: str):
    """
    Filters out lines from a BED file that have negative values in
    the second or third column.
    """
    with open(input_path, "r") as infile:
        lines = infile.readlines()
    with open(output_path, "w") as outfile:
        for line_number, line in enumerate(lines, start=0):
            cols = line.strip().split("\t")
            if int(cols[1]) < 0 or int(cols[2]) < 0:
                print(f"Negative coordinate found on line: {line_number}. Skipping.")
            else:
                outfile.write(line)


def filter_bed_chrom_regions(input_path: str, output_path: str, chrom_sizes_file: str):
    """
    Filters out lines from a BED file that are out of bounds of the chromosome size.
    """
    filtered_lines = []
    total_lines = 0
    chrom_sizes = {}

    with open(chrom_sizes_file, "r") as sizes:
        for line in sizes:
            chrom, size = line.strip().split("\t")[0:2]
            size = int(size)
            chrom_sizes[chrom] = size

    with open(input_path, "r") as infile:
        lines = infile.readlines()
    with open(output_path, "w") as outfile:
        for bed_line in lines:
            total_lines += 1
            bed_cols = bed_line.strip().split("\t")
            chrom = bed_cols[0]
            if chrom in chrom_sizes and int(bed_cols[2]) <= chrom_sizes[chrom]:
                filtered_lines.append(bed_line)

    # Sort the filtered BED file
    sorted_lines = sorted(
        filtered_lines, key=lambda x: (x.split("\t")[0], int(x.split("\t")[1]))
    )

    with open(output_path, "w") as outfile:
        outfile.writelines(sorted_lines)

    print(f"chrom size: removed {total_lines - len(sorted_lines)}/{total_lines} lines.")


def get_regions_from_bed(regions_bed_filename: str):
    """
    Read BED file and yield a region (chrom, start, end) for each invocation.
    """
    with open(regions_bed_filename, "r") as fh_bed:
        for line in fh_bed:
            line = line.rstrip("\r\n")

            if line.startswith("#"):
                continue

            columns = line.split("\t")
            chrom = columns[0]
            start, end = [int(x) for x in columns[1:3]]
            region = chrom, start, end
            yield region


def filter_bed_on_idx(regions_bed_filename: str, idx: np.ndarray):
    """
    Keeps regions in a BED file that are on the given index.
    """
    idx = set(idx)

    with open(regions_bed_filename, "r") as infile:
        lines = infile.readlines()
    with open(regions_bed_filename, "w") as outfile:
        for i, line in enumerate(lines):
            if i in idx:
                cols = line.strip().split()
                outfile.write("\t".join(cols) + "\n")


def get_bed_region_width(regions_bed_filename: str):
    """Get the width of a region in a BED file."""
    with open(regions_bed_filename, "r") as bed_file:
        first_line = bed_file.readline()
    fields = first_line.split("\t")
    start_position = int(fields[1])
    end_position = int(fields[2])
    return end_position - start_position


def fix_bed_labels(regions_bed_file: str):
    """
    Fix the labels of a BED file (3rd column) so that they are in the format
    chr:start-end.
    """
    with open(regions_bed_file, "r") as infile:
        lines = infile.readlines()
    with open(regions_bed_file, "w") as outfile:
        for line in lines:
            cols = line.strip().split()
            if len(cols) == 3:
                cols.append(f"{cols[0]}:{cols[1]}-{cols[2]}")
            elif len(cols) >= 4:
                cols[3] = f"{cols[0]}:{cols[1]}-{cols[2]}"
            if len(cols) > 4:
                cols = cols[:4]
            outfile.write("\t".join(cols) + "\n")
