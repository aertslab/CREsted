import gc
import os
from collections.abc import Sequence
from copy import copy
from os import PathLike

import numpy as np
import pybigtools
import tqdm
from loguru import logger
from scipy.sparse import csr_array

from crested._genome import Genome
from crested._io import _extract_tracks_from_bigwig


class TrackData:
    def __init__(
        self,
        paths: dict[str] | list[str] | str | PathLike,
        bin_size: int = 1,
        target: str = "mean",
        crop: int | tuple[int, int] = 0,
        in_memory: bool = True,
        chrom_sizes: dict[str, int] | Genome | None = None,
        compressed: bool = False,
        verbose: bool = False
    ):
        """
        A class to extract track data from multiple BigWigs at once. Can extract either on the fly (in_memory=False) or read in all bigwigs (in_memory=True).

        Parameters
        ----------
        paths
            A list or dict of paths, or a path to a directory. If a dict, keys are kept as class_names. If a list or directory, class_names are inferred from the filenames.
        bin_size
            Size to bin the output values to, in basepairs. If None, doesn't do binning.
        target
            The manner in which to pool values. Can be 'mean', 'max', 'count', or 'logcount'
        crop
            A single int or tuple of ints, denoting nucleotides to crop from the sides of regions before reading out the track.
            A single value will remove half that from both flank, while a tuple will remove those amounts from either side.
            Example: returning center 500bp tracks from 1000bp regions works with crop=500 or crop=(250, 250).
        in_memory
            Whether to save the bigwig values in memory, or read out tracks on the fly. If doing actual training, recommended to turn on, 
            since on-the-fly track retrieval is likely not fast enough to keep up. Default is True.
        chrom_sizes
            The chromosomes to read in with their sizes, should be exactly the same between datasets.
            The keys in this determine which chromosomes are read in; if you want to skip i.e. the mitochondrial genome, drop it here.
            Can be a dict, a crested.Genome object, or None. If None (default), read in from the first bigwig listed.
            Note that TrackData handles mixed presence/absence 'chr' prefixes by default when loading in data.
        compressed
            Whether to save the data as compressed sparse rows. Currently not recommended since this increases memory usage peaks 
            while reading in data, while not resulting in large gains in terms of stable memory occupancy. Default is False.
        verbose
            Whether to print extra information when reading in data. Default is False.
        """
        # If paths is a single directory, read in files
        if isinstance(paths, str) or isinstance(paths, PathLike):
            paths = [file for file in os.listdir(paths) if os.path.isfile(os.path.join(paths, file))]

        # Check and save paths and path names
        if isinstance(paths, Sequence):
            self.paths = paths
            self.class_names = [os.path.splitext(os.path.basename(path))[0] for path in paths]
        else:
            self.paths = list(paths.values())
            self.class_names = list(paths.keys())

        # Save cropping size
        if isinstance(crop, Sequence):
            assert len(crop) == 2, "If providing a list/tuple to `crop`, needs to be length 2."
            self.crop = crop
        else:
            assert crop % 2 == 0, "If providing a single cropping integer, must be divisible by 2, since we remove crop//2 from both flanks."
            self.crop = (crop // 2, crop // 2)


        # Save other arguments
        self.bin_size = None if bin_size == 1 else bin_size
        self.target = target
        self.in_memory = in_memory
        self.chrom_sizes = chrom_sizes.chrom_sizes if isinstance(chrom_sizes, Genome) else chrom_sizes
        self.compressed = compressed
        self.verbose = verbose

        # Do initial bigwig checks and housekeeping
        logger.info("Checking whether bigwigs exist")
        self.chrom_name_mapping = {}
        for path in self.paths:
            with pybigtools.open(path) as f:
                # Check if file is actually a bigwig
                assert f.is_bigwig, f"{path} does not appear to be a bigwig file."
                # If no chromsizes provided, use from the first bigwig we see
                if self.chrom_sizes is None:
                    self.chrom_sizes = copy(f.chroms())
                # Do some fuzzy matching to work with both bigwigs using chr prefix and no chr prefix
                ## Get definitely prefix-less ver to compare and map back to original chromnames. (i.e '3': '3' or '3': 'chr3', depending on this bw's chromosomes)
                curr_bw_prefixless2orig = {curr_bw_chrom.replace('chr', ''): curr_bw_chrom for curr_bw_chrom in f.chroms()}
                ## Then use the prefixless ver of the ref chroms to link the original ref chrom to the original bw chrom
                self.chrom_name_mapping[path] = {ref_chrom: curr_bw_prefixless2orig[ref_chrom.replace('chr', '')] for ref_chrom in self.chrom_sizes}

        # Load regions into memory
        if self.in_memory:
            logger.info("Reading bigwigs into memory")
            self._read_bws()
            logger.info("Finished reading bigwigs into memory")

        # TODO maybe: add cropping function to return center part of submitted region (but maybe is more for the custom dataloader to shrink input seq there)?
        # TODO maybe: add optional contigs argument, where we only read in values from specific regions rather than from everywhere.
        #   Small problem: augmentation might shift boundaries, expect ppl to handle this beforehand?
        #   Also: how much does this actually save? Idea is for only loading regions around genes, but eh
        # TODO maybe: add support for stranded bigwigs (provide pair of paths, save both in alternative version, but bit of a bother to implement)

    def __getitem__(self, idx):
        """"""
        # Parse string index
        if isinstance(idx, str):
            n_colons = idx.count(':')
            if n_colons == 1:
                chrom, start_end = idx.split(':')
                strand = "+"
            elif n_colons == 2:
                chrom, start_end, strand = idx.split(':')
            else:
                raise ValueError(f"Expect either 1 or 2 colons (chr:start-end or chr:start-end:split), not like this: {idx}")
            start, end = map(int, start_end.split('-'))
        # Parse tuple index
        else:
            chrom = idx[0]
            start = idx[1]
            end = idx[2]
            if len(idx) >= 4:
                strand = idx[3]
            else:
                strand = "+"

        if self.crop[0] > 0 or self.crop[1] > 0:
            start = start + self.crop[0]
            end = end - self.crop[1]

        # Get values from saved values or by reading in on the fly
        if self.in_memory:
            assert chrom in self.data, f"Could not find chromosome {chrom} in read-in data (chromosomes {list(self.data.keys())})"
            if self.compressed:
                values = self.data[chrom][start:end, :].toarray()
            else:
                values = self.data[chrom][start:end, :]
        else:
            values = self._get_single_region(chrom, start, end, strand)

        # Convert to float32 (pybigtools returns float64 by default, while our in_memory bws are in float16 to save memory)
        values = values.astype('float32')

        # Do binning
        if self.bin_size > 1:
            if self.target == 'mean':
                values = values.reshape((-1, self.bin_size, values.shape[-1])).mean(axis=1)
            elif self.target == 'max':
                values = values.reshape((-1, self.bin_size, values.shape[-1])).max(axis=1)
            elif self.target == 'count' or self.target == 'sum':
                values = values.reshape((-1, self.bin_size, values.shape[-1])).sum(axis=1)
            elif self.target == 'logcount' or self.target == 'logsum':
                values = np.log1p(values.reshape((-1, self.bin_size, values.shape[-1])).sum(axis=1))

        return values

    def _read_bws(self):
        """"""
        # Initialize empty arrays per chromosome
        self.data = {chrom_name: np.zeros((chrom_size, len(self.paths)), dtype='float16') for chrom_name, chrom_size in self.chrom_sizes.items()}

        # Read in data per path
        if self.verbose:
            logger.info("TEMP: Opening bw files")
        file_objects = {path: pybigtools.open(path, "r") for path in self.paths}

        # TODO: parallelize? but likely have to go back to per-file first
        # Read in one chrom at a time so that we can compress the entire chrom readout before moving to the next one
        if self.verbose:
            logger.info("TEMP: Starting chrom reading loop")
        try:
            for chrom in tqdm.tqdm(self.chrom_sizes):
                for path_i, bw_path in enumerate(self.paths):
                    # Extract values for this chromosome per opened bw
                    if self.verbose:
                        logger.info(f"TEMP: Reading chrom {chrom} from path {bw_path}")
                    values = file_objects[bw_path].values(
                        self.chrom_name_mapping[bw_path][chrom], # Get the chrom name in this specific bigwig
                        exact=True,
                        missing=0.0,
                        oob=0.0,
                    )
                    values = values.astype('float16')
                    self.data[chrom][:, path_i] = values

                # Convert values to column sparse arrays
                if self.compressed:
                    if self.verbose:
                        logger.info(f"TEMP: Converting chrom {chrom} to csc")
                    self.data[chrom] = csr_array(self.data[chrom])
                if self.verbose:
                    logger.info("TEMP: Collecting garbage")
                gc.collect()
        finally:
            for path in file_objects:
                file_objects[path].close()

    def _get_single_region(self, chrom, start, end, strand = "+"):
        """"""
        values = np.concatenate([
            _extract_tracks_from_bigwig(
                bw_file = bw_path,
                coordinates = [(self.chrom_name_mapping[bw_path][chrom], start, end)],
            ) for bw_path in self.paths
        ], axis = 0).T
        if strand == "-":
            values = np.flip(values, axis=0) # TODO: test
        # TODO: implement binning through pybigtools directly
        return values

    @property
    def obs_names(self):
        return self.class_names

    def __len__(self):
        return len(self.paths)
