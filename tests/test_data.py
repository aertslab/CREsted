"""Pytest for data module."""

from contextlib import contextmanager

import pytest
from loguru import logger


@contextmanager
def log_capture(level="WARNING"):
    messages = []
    handler_id = logger.add(messages.append, level=level)
    try:
        yield messages
    finally:
        logger.remove(handler_id)


def test_genome_persistence(genome_path):
    """Test that the genome object is correctly stored."""
    import crested

    # check that does not yet exist
    assert crested._conf.genome is None

    genome = crested.Genome(
        fasta=genome_path,
        chrom_sizes={"chr1": 1000, "chr2": 2000},
    )
    crested.register_genome(genome)

    # check that it is stored
    assert crested._conf.genome == genome


def test_no_genome_fasta():
    """Test that a TypeError is raised if no fasta is provided."""
    import crested

    with pytest.raises(TypeError):
        crested.Genome(
            chrom_sizes={"chr1": 1000, "chr2": 2000}, annotation="tests/data/test.gtf"
        )


def test_import_beds_with_genome(genome_path):
    """Test that import_beds uses genome chromsizes."""
    import crested

    # Scenario 1: Genome registered with chromsizes provided
    with log_capture(level="WARNING") as messages:
        genome = crested.Genome(
            fasta=genome_path,
            chrom_sizes="tests/data/test.chrom.sizes",
        )
        crested.register_genome(genome)
        crested.import_beds(
            beds_folder="tests/data/test_topics",
            regions_file="tests/data/test.regions.bed",
        )

    warning_messages = list(messages)
    warning_text_chromsizes = (
        "Chromsizes file not provided. Will not check if regions are within chromosomes"
    )
    warning_text_filtered = "Filtered 1 consensus regions (not within chromosomes)"
    assert all(
        warning_text_chromsizes not in msg for msg in warning_messages
    ), "Warning about missing chromsizes was unexpectedly raised."
    assert any(
        warning_text_filtered in msg for msg in warning_messages
    ), "Expected warning about filtered regions was not raised."

    # Scenario 2: Chromsizes provided via parameter, no genome registered
    crested._conf.genome = None
    with log_capture(level="WARNING") as messages:
        crested.import_beds(
            beds_folder="tests/data/test_topics",
            regions_file="tests/data/test.regions.bed",
            chromsizes_file="tests/data/test.chrom.sizes",
        )

    warning_messages = list(messages)
    assert all(
        warning_text_chromsizes not in msg for msg in warning_messages
    ), "Warning about missing chromsizes was unexpectedly raised."
    assert any(
        warning_text_filtered in msg for msg in warning_messages
    ), "Expected warning about filtered regions was not raised."

    # Scenario 3: No chromsizes provided via genome or parameter
    with log_capture(level="WARNING") as messages:
        crested.import_beds(
            beds_folder="tests/data/test_topics",
            regions_file="tests/data/test.regions.bed",
            chromsizes_file=None,  # Explicitly set to None
        )

    warning_messages = list(messages)
    assert any(
        warning_text_chromsizes in msg for msg in warning_messages
    ), "Expected warning about missing chromsizes was not raised."
    assert all(
        warning_text_filtered not in msg for msg in warning_messages
    ), "Warning about filtered regions was unexpectedly raised."


def test_genome_fetch(genome_path):
    """Test reading the genome."""
    import crested

    genome = crested.Genome(
        fasta=genome_path,
        chrom_sizes="tests/data/test.chrom.sizes",
    )
    seq = genome.fetch("chr1", 10000, 10100)
    assert len(seq) == 100


def test_genome_fetch_region(genome_path):
    """Test reading the genome with a region string."""
    import crested

    genome = crested.Genome(
        fasta=genome_path,
        chrom_sizes="tests/data/test.chrom.sizes",
    )
    seq1 = genome.fetch("chr1", 10000, 10100)
    seq2 = genome.fetch(region="chr1:10000-10100")
    assert seq1 == seq2


def test_genome_fetch_reverse(genome_path):
    """Test reading the genome on the negative strand."""
    import crested

    genome = crested.Genome(
        fasta=genome_path,
        chrom_sizes="tests/data/test.chrom.sizes",
    )
    seq_forward = genome.fetch("chr1", 10000, 10100)
    seq_rev = genome.fetch("chr1", 10000, 10100, "-")
    seq_rev_region = genome.fetch(region="chr1:10000-10100:-")
    assert seq_rev == crested.utils.reverse_complement(seq_forward)
    assert seq_rev_region == seq_rev


def test_genome_fetch_mismatch(genome_path):
    """Test reading the genome when supplying both coordinates and a region."""
    import crested

    genome = crested.Genome(
        fasta=genome_path,
        chrom_sizes="tests/data/test.chrom.sizes",
    )
    seq = genome.fetch("chr1", 10000, 10100, region="chr1:10000-10200")
    assert len(seq) == 100


def test_genome_fetch_missing(genome_path):
    """Test reading the genome when not supplying all information"""
    import crested

    genome = crested.Genome(
        fasta=genome_path,
        chrom_sizes="tests/data/test.chrom.sizes",
    )
    with pytest.raises(ValueError):
        genome.fetch("chr1", 10000)
    with pytest.raises(ValueError):
        genome.fetch("chr1", end=10100)
    with pytest.raises(ValueError):
        genome.fetch("chr1", 10000, region="chr1:10000-10200")
