"""Genome object for storing information about a genome and genome registry."""

from __future__ import annotations

from pathlib import Path

from crested import _conf as conf


class Genome:
    """
    A class that encapsulates information about a genome, including its FASTA sequence, its annotation, and chromosome sizes.

    Adapted from https://github.com/kaizhang/SnapATAC2/blob/main/snapatac2-python/python/snapatac2/genome.py.

    Attributes
    ----------
    fasta
        The path to the FASTA file.
    chrom_sizes
        A path to a tab delimited chromsizes file or a dictionary containing chromosome names and sizes.
    annotation
        The path to the annotation file.
    """

    def __init__(
        self,
        *,
        fasta: Path,
        chrom_sizes: dict[str, int] | Path | None = None,
        annotation: Path | None = None,
    ):
        """Initialize the Genome object."""
        if isinstance(fasta, Path) or isinstance(fasta, str):
            self._fasta = Path(fasta)
        else:
            raise ValueError("fasta must be a Path.")
        self._annotation = annotation
        if isinstance(chrom_sizes, str) or isinstance(chrom_sizes, Path):
            self._chrom_sizes = Path(chrom_sizes)
        else:
            self._chrom_sizes = chrom_sizes

    @property
    def fasta(self) -> Path:
        """
        The Path to the FASTA file.

        Returns
        -------
        The path to the FASTA file.
        """
        return self._fasta

    @property
    def annotation(self) -> Path | None:
        """
        The Path to the annotation file.

        Returns
        -------
        The path to the annotation file.
        """
        return self._annotation

    @property
    def chrom_sizes(self) -> dict[str, int]:
        """
        A dictionary with chromosome names as keys and their lengths as valeus.

        Returns
        -------
        A dictionary of chromosome sizes.
        """
        if self._chrom_sizes is None:
            from pysam import FastaFile

            fasta = FastaFile(self.fasta)
            self._chrom_sizes = dict(zip(fasta.references, fasta.lengths))
        elif isinstance(self._chrom_sizes, Path):
            from crested._io import _read_chromsizes

            self._chrom_sizes = _read_chromsizes(self._chrom_sizes)
        elif not isinstance(self._chrom_sizes, dict):
            print(self._chrom_sizes)
            raise ValueError("chrom_sizes must be a dictionary or a Path.")
        return self._chrom_sizes


def register_genome(genome: Genome):
    """
    Register a genome to be used throughout a session.

    Once a genome is registered, all the functions in the package that require a genome will use it if not explicitly provided.

    Parameters
    ----------
    genome
        The Genome object to register.
    """
    if not isinstance(genome, Genome):
        raise TypeError("genome must be an instance of Genome")
    conf.genome = genome
