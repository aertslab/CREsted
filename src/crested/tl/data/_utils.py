from __future__ import annotations

import re


def _flip_region_strand(region: str) -> str:
    """Reverse the strand of a region."""
    strand_reverser = {"+": "-", "-": "+"}
    return region[:-1] + strand_reverser[region[-1]]


def _check_region_strandedness(region: str) -> bool:
    """Check the strandedness of a region, raising an error if the formatting isn't recognised."""
    if re.fullmatch(r".+:\d+-\d+:[-+]", region):
        return True
    elif re.fullmatch(r".+:\d+-\d+", region):
        return False
    else:
        raise ValueError(
            f"Region {region} was not recognised as a valid coordinate set (chr:start-end or chr:start-end:strand)."
            "If provided, strand must be + or -."
        )

def _split_region(region: str) -> tuple[str, int, int, str]:
    """Split a region string in constituent chr, start, end and strand.

    If strand is not provided, infers + to keep output consistent.
    """
    try:
        chrom, start_end, strand = region.split(':')
    except ValueError:
        try:
            chrom, start_end = region.split(':')
            strand = "+"
        except ValueError as err:
            raise ValueError(f"Expect region with pattern chr:start-end:strand or chr:start-end, not {region}") from err
    start, end = map(int, start_end.split('-'))
    return chrom, start, end, strand
