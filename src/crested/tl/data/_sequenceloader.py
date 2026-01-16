from __future__ import annotations

import keras
from loguru import logger
from tqdm import tqdm

if keras.config.backend() == "torch":
    import torch
    FrameworkDatasetClass = torch.utils.data.Dataset
else:
    FrameworkDatasetClass = object

from crested._genome import Genome

from ._utils import _check_region_strandedness, _split_region


class SequenceLoader:
    """
    Load sequences from a genome file.

    Options for stochastic shifting are available.

    Parameters
    ----------
    genome
        Genome instance.
    in_memory
        If True, the sequences of supplied regions will be loaded into memory.
    max_stochastic_shift
        Maximum stochastic shift (n base pairs) to apply randomly to each sequence.
    regions
        List of (expanded/augmented) regions to load into memory. Required if in_memory is True.
    always_reverse_complement, deterministic_shift
        Deprecated arguments, to be handled in your DataWrapper/AnnDataset classes instead.
    """

    def __init__(
        self,
        genome: Genome,
        in_memory: bool = False,
        max_stochastic_shift: int = 0,
        regions: list[str] | None = None,
        always_reverse_complement = 'deprecated',
        deterministic_shift = 'deprecated',
    ):
        """Initialize the SequenceLoader with the provided genome file and options."""
        if always_reverse_complement != 'deprecated':
            logger.warning("always_reverse_complement in SequenceLoader is deprecated - please handle expanding indices with rev-comp'd version in the wrapping class and pass pre-expanded regions to `regions` instead.")
        if deterministic_shift != 'deprecated':
            logger.warning("deterministic_shift in SequenceLoader is deprecated - please handle expanding indices with deterministic shifts in the wrapping class and pass pre-expanded regions to `regions` instead.")
        self.genome = genome.fasta
        self.chromsizes = genome.chrom_sizes
        self.in_memory = in_memory
        self.max_stochastic_shift = max_stochastic_shift
        self.sequences = {}
        self.complement = str.maketrans("ACGT", "TGCA")
        self.regions = regions

        if self.in_memory:
            self._load_sequences_into_memory(self.regions)
        # TODO: maybe add check for sequence length

    def _load_sequences_into_memory(self, regions: list[str]):
        """Load all sequences into memory (dict)."""
        logger.info("Loading sequences into memory...")
        # Check region formatting
        stranded = _check_region_strandedness(regions[0])

        for region in tqdm(regions):
            # Make region stranded if not
            if not stranded:
                strand = "+"
                region = f"{region}:{strand}"
                if region[-4] == ":":
                    raise ValueError(
                        f"You are double-adding strand ids to your region {region}. Check if all regions are stranded or unstranded."
                    )

            # Parse region
            chrom, start, end, strand = _split_region(region)

            # Add region to self.sequences
            extended_sequence = self._get_extended_sequence(
                chrom, start, end, strand
            )
            self.sequences[region] = extended_sequence

    def _get_extended_sequence(
        self, chrom: str, start: int, end: int, strand: str
    ) -> str:
        """Get sequence from genome file, extended for stochastic shifting."""
        extended_start = start - self.max_stochastic_shift
        extended_end = end + self.max_stochastic_shift
        padding_start, padding_end = 0, 0

        if extended_start < 0:
            padding_start = -extended_start
            extended_start = 0
        if self.chromsizes and (extended_end > self.chromsizes[chrom]):
            padding_end = extended_end - self.chromsizes[chrom]
            extended_end = self.chromsizes[chrom]

        seq = self.genome.fetch(chrom, extended_start, extended_end).upper()
        if padding_start > 0:
            seq = "N"*padding_start + seq
        if padding_end > 0:
            seq = seq + "N"*padding_end

        if strand == "-":
            seq = self._reverse_complement(seq)
        return seq

    def _reverse_complement(self, sequence: str) -> str:
        """Reverse complement a sequence."""
        return sequence.translate(self.complement)[::-1]

    def get_sequence(
        self, region: str, stranded: bool | None = None, shift: int = 0
    ) -> str:
        """
        Get sequence for a region, strand, and shift from memory or fasta.

        If no strand is given in region or strand, assumes positive strand.

        Parameters
        ----------
        region
            Region to get the sequence for. Either (chr:start-end) or (chr:start-end:strand).
        stranded
            Whether the input data is stranded. Default (None) infers from sequence (at a computational cost).
            If not stranded, positive strand is assumed.
        shift:
            Shift of the sequence within the extended sequence, for use with the stochastic shift mechanism.

        Returns
        -------
        The DNA sequence, as a string.
        """
        if stranded is None:
            stranded = _check_region_strandedness(region)
        if not stranded:
            region = f"{region}:+"
        # Parse region
        chrom, start, end, strand = _split_region(region)
        chrom_size = self.chromsizes[chrom] if self.chromsizes is not None else None

        # Check if within genomic boundaries, clip otherwise
        defacto_shift = -shift if strand == "-" else shift
        overhang = 0
        if start + defacto_shift < 0:
            overhang = start + defacto_shift
        elif chrom_size is not None and (end + defacto_shift) > chrom_size:
            overhang = end + defacto_shift - chrom_size
        defacto_shift -= overhang
        shift = -defacto_shift if strand == "-" else defacto_shift

        # Get extended sequence
        if self.in_memory:
            sequence = self.sequences[region]
        else:
            sequence = self._get_extended_sequence(chrom, start, end, strand)

        # Extract from extended sequence
        start_idx = self.max_stochastic_shift + shift
        end_idx = start_idx + (end - start)
        sub_sequence = sequence[start_idx:end_idx]

        # Pad with Ns if sequence is shorter than expected
        if len(sub_sequence) < (end - start):
            if strand == "+":
                sub_sequence = sub_sequence.ljust(end - start, "N")
            else:
                sub_sequence = sub_sequence.rjust(end - start, "N")

        return sub_sequence



