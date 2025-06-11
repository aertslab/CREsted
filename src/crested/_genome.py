"""Genome object for storing information about a genome and genome registry."""

from __future__ import annotations

import errno
import os
from functools import cached_property
from pathlib import Path

from loguru import logger
from pysam import FastaFile

import crested._conf as conf
from crested.utils._seq_utils import reverse_complement


class Genome:
    """
    A class that encapsulates information about a genome, including its FASTA sequence, its annotation, and chromosome sizes.

    Adapted from https://github.com/kaizhang/SnapATAC2/blob/main/snapatac2-python/python/snapatac2/genome.py.

    Parameters
    ----------
    fasta
        The path to the FASTA file.
    chrom_sizes
        A path to a tab delimited chromsizes file or a dictionary containing chromosome names and sizes.
        If not provided, the chromosome sizes will be inferred from the FASTA file.
    annotation
        The path to the annotation file.
    name
        Optional name of the genome.

    Examples
    --------
    >>> genome = Genome(
    ...     fasta="tests/data/test.fa",
    ...     chrom_sizes="tests/data/test.chrom.sizes",
    ... )
    >>> print(genome.fasta)
    <pysam.libcfaidx.FastaFile at 0x7f4d8b4a8f40>
    >>> print(genome.chrom_sizes)
    {'chr1': 1000, 'chr2': 2000}
    >>> print(genome.name)
    test

    See Also
    --------
    crested.register_genome
    """

    def __init__(
        self,
        fasta: Path,
        chrom_sizes: dict[str, int] | Path | None = None,
        annotation: Path | None = None,
        name: str | None = None,
    ):
        """Initialize the Genome object."""
        if isinstance(fasta, (Path, str)):
            fasta = Path(fasta)
            if not os.path.exists(fasta):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), str(fasta)
                )
            self._fasta = fasta
        else:
            raise ValueError("fasta must be a Path.")

        if isinstance(chrom_sizes, (Path, str)):
            chrom_sizes = Path(chrom_sizes)
            if not os.path.exists(chrom_sizes):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), str(chrom_sizes)
                )
            self._chrom_sizes = chrom_sizes
        else:
            self._chrom_sizes = chrom_sizes

        if isinstance(annotation, (Path, str)):
            annotation = Path(annotation)
            if not os.path.exists(annotation):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), str(annotation)
                )
            self._annotation = annotation
        else:
            self._annotation = None

        self._name = name

    @cached_property
    def fasta(self) -> FastaFile:
        """
        The pysam FastaFile object for the FASTA file.

        Returns
        -------
        The pysam FastaFile object.
        """
        return FastaFile(self._fasta)

    @property
    def annotation(self) -> Path | None:
        """
        The Path to the annotation file.

        Currently not used in the package.

        Returns
        -------
        The path to the annotation file.
        """
        return self._annotation

    @property
    def chrom_sizes(self) -> dict[str, int]:
        """
        A dictionary with chromosome names as keys and their lengths as values.

        Returns
        -------
        A dictionary of chromosome sizes.
        """
        if self._chrom_sizes is None:
            self._chrom_sizes = dict(zip(self.fasta.references, self.fasta.lengths))
        elif isinstance(self._chrom_sizes, Path):
            from crested._io import _read_chromsizes

            self._chrom_sizes = _read_chromsizes(self._chrom_sizes)
        elif not isinstance(self._chrom_sizes, dict):
            raise ValueError("chrom_sizes must be a dictionary or a Path.")
        return self._chrom_sizes

    @property
    def name(self) -> str:
        """
        The name of the genome.

        Returns
        -------
        The name of the genome.
        """
        if self._name is None:
            filename = self.fasta.filename.decode("utf-8")
            basename = os.path.basename(filename)

            if basename.endswith(".fa") or basename.endswith(".fasta"):
                return os.path.splitext(basename)[0]  # Remove the extension and return
            else:
                return basename
        return self._name

    def fetch(
        self,
        chrom: str | None = None,
        start: int | None = None,
        end: int | None = None,
        strand: str = "+",
        region: str | None = None,
    ) -> str:
        """
        Fetch a sequence from a genomic region.

        Start and end denote 0-based, half-open intervals, following the bed convention.

        Parameters
        ----------
        chrom
            The chromosome of the region to extract.
        start
            The start of the region to extract. Assumes 0-indexed positions.
        end
            The end of the region to extract, exclusive.
        strand
            The strand of the region. If '-', the sequence is reverse-complemented. Default is "+".
        region
            Alternatively, a region string to parse. If supplied together with chrom/start/end, explicit coordinates take priority.

        Returns
        -------
        The requested sequence, as a string.
        """
        if region and (chrom or start or end):
            logger.warning(
                "Both region and chrom/start/end supplied. Using chrom/start/end..."
            )
        elif region:
            if region[-2] == ":":
                chrom, start_end, strand = region.split(":")
            else:
                chrom, start_end = region.split(":")
            start, end = map(int, start_end.split("-"))

        if not (chrom and start and end):
            raise ValueError(
                "chrom/start/end must all be supplied to extract a sequence."
            )

        seq = self.fasta.fetch(reference=chrom, start=start, end=end)
        if strand == "-":
            return reverse_complement(seq)
        else:
            return seq

    def __repr__(self) -> str:
        """Return a string representation of the Genome object."""
        fasta_exists = self.fasta is not None
        chrom_sizes_exists = self.chrom_sizes is not None
        annotations_exists = self.annotation is not None
        return f"Genome({self.name}, fasta={fasta_exists}, chrom_sizes={chrom_sizes_exists}, annotation={annotations_exists})"


def register_genome(genome: Genome):
    """
    Register a genome to be used throughout a session.

    Once a genome is registered, all the functions in the package that require a genome will use it if not explicitly provided.

    Parameters
    ----------
    genome
        The Genome object to register.

    Examples
    --------
    >>> genome = Genome(
    ...     fasta="tests/data/hg38.fa",
    ...     chrom_sizes="tests/data/test.chrom.sizes",
    ... )
    >>> register_genome(genome)
    INFO Genome hg38 registered.
    """
    if not isinstance(genome, Genome):
        raise TypeError("genome must be an instance of crested.Genome")
    conf.genome = genome
    logger.info(f"Genome {genome.name} registered.")


def _resolve_genome(
    genome: os.PathLike | Genome | None,
    chromsizes_file: os.PathLike | None = None,
    annotation: os.PathLike | None = None,
) -> Genome:
    """Resolve the input to a Genome object. Required to keep backwards compatibility with fasta and chromsizes paths as inputs."""
    if isinstance(genome, Genome):
        return genome
    if genome is not None:
        genome_path = Path(genome)
        if not genome_path.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(genome_path)
            )
        return Genome(
            fasta=genome_path, chrom_sizes=chromsizes_file, annotation=annotation
        )
    else:
        if conf.genome is not None:
            return conf.genome
        else:
            raise ValueError("No genome provided or registered.")
