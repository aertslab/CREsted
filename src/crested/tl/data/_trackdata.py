import gc
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from os import PathLike
from typing import Literal

import numpy as np
import pybigtools
import tqdm
from loguru import logger
from scipy.sparse import csr_array

import crested._conf as conf
from crested._io import _extract_tracks_from_bigwig
from crested.utils import parse_region


class TrackData:
    def __init__(
        self,
        paths: dict[str] | list[str] | str | PathLike,
        bin_size: int = 1,
        target: Literal['mean', 'max', 'count', 'logcount'] = "mean",
        prebinned: bool = False,
        crop: int | tuple[int, int] = 0,
        in_memory: bool = True,
        kept_chroms: list[str] | None = None,
        chrom_sizes: dict[str, int] | None = None,
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
        prebinned
            Whether to save the dataset in binned mode (limiting augmentation to bin_size shifts but reducing memory usage)
            or at base-pair resolution (allowing extraction of any sequence, but increasing memory usage)
        crop
            A single int or tuple of ints, denoting nucleotides to crop from the sides of regions before reading out the track.
            A single value will remove half that from both flank, while a tuple will remove those amounts from either side.
            Example: returning center 500bp tracks from 1000bp regions works with crop=500 or crop=(250, 250).
        in_memory
            Whether to save the bigwig values in memory, or read out tracks on the fly. If doing actual training, recommended to turn on,
            since on-the-fly track retrieval is likely not fast enough to keep up. Default is True.
        kept_chroms
            Chromosomes to actually read in. if you want to skip i.e. the mitochondrial genome, drop it here.
            Note that TrackData handles mixed presence/absence 'chr' prefixes by default when loading in data.
        chrom_sizes
            Chromosome sizes. Should be exactly the same between datasets.
            Can be a dict or None. If None (default), uses the registered genome if available, and otherwise read in from the first bigwig listed.
        compressed
            Whether to save the data as compressed sparse rows. Currently not recommended since this increases memory usage peaks
            while reading in data, while not resulting in large gains in terms of stable memory occupancy. Default is False.
        verbose
            Whether to print extra information when reading in data. Default is False.
        """
        # Save simple arguments
        self.bin_size = 1 if bin_size is None else bin_size
        self.prebinned = prebinned
        self.target = target
        self.in_memory = in_memory
        self.compressed = compressed
        self.verbose = verbose

        if self.prebinned and self.bin_size == 1:
            raise ValueError("'prebinned' cannot be True if not binning (bin_size is 1/None).")

        # If paths is a single directory, gather files
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

        # If no chromsizes provided, use registered genome or otherwise the first bigwig we see
        if chrom_sizes is not None:
            self.chrom_sizes = chrom_sizes
        else:
            if conf.genome is not None:
                self.chrom_sizes = conf.genome.chrom_sizes
            else:
                with pybigtools.open(self.paths[0]) as f:
                    self.chrom_sizes = copy(f.chroms())

        # Filter chrom_sizes to kept_chroms
        if kept_chroms is not None:
            for chrom in kept_chroms:
                if chrom not in self.chrom_sizes:
                    raise KeyError(f"Could not find chromosome {chrom} in chrom_sizes.\nCurrent chrom_sizes chroms: {list(self.chrom_sizes.keys())}")
            self.chrom_sizes = {chrom: chrom_size for chrom, chrom_size in self.chrom_sizes.items() if chrom in kept_chroms}

        if self.prebinned:
            # round down to nearest multiple of bin_size
            self.binned_chrom_sizes = {chrom: self.chrom_sizes[chrom]//self.bin_size for chrom in self.chrom_sizes}

        # Do bigwig checks and fuzzy chromosome name matching
        logger.info("Checking whether bigwigs exist")
        self.chrom_name_mapping = {}
        for path in self.paths:
            with pybigtools.open(path) as f:
                # Check if file is actually a bigwig
                assert f.is_bigwig, f"{path} does not appear to be a bigwig file."
                # Do some fuzzy matching to work with both bigwigs using chr prefix and no chr prefix
                ## Get definitely prefix-less ver to compare and map back to original chromnames. (i.e '3': '3' or '3': 'chr3', depending on this bw's chromosomes)
                curr_bw_prefixless2orig = {curr_bw_chrom.replace('chr', ''): curr_bw_chrom for curr_bw_chrom in f.chroms()}
                ## Then use the prefixless ver of the ref chroms to link the original ref chrom to the original bw chrom
                ## i.e. {'path1': {'chr3': '3', ...}, 'path2': {'chr3': 'chr3', ...}, ...} if chrom_sizes is using 'chr3'
                self.chrom_name_mapping[path] = {ref_chrom: curr_bw_prefixless2orig[ref_chrom.replace('chr', '')] for ref_chrom in self.chrom_sizes}

        # Load regions into memory
        if self.in_memory:
            logger.info("Reading bigwigs into memory")
            self._read_bws()
            logger.info("Finished reading bigwigs into memory")

        # TODO maybe: add optional contigs argument, where we only read in values from specific regions rather than from everywhere.
        #   Small problem: augmentation might shift boundaries, expect ppl to handle this beforehand?
        #   Also: how much does this actually save? Idea is for only loading regions around genes, but eh
        # TODO maybe: add support for stranded bigwigs (provide pair of paths, save both in alternative version, but bit of a bother to implement)
        # TODO: implement region reading rather than full chromosomes

    def __getitem__(self, idx: tuple[str, int, int] | tuple[str, int, int, str] | str):
        return self.get_track(idx, shift=0)

    def get_track(self, idx: tuple[str, int, int] | tuple[str, int, int, str] | str, shift: int = 0):
        """"""
        # Parse index
        chrom, start, end, strand = parse_region(idx)

        if self.crop[0] > 0 or self.crop[1] > 0:
            start = start + self.crop[0]
            end = end - self.crop[1]

        # Shift
        real_shift = -shift if strand == "-" else shift
        start += real_shift
        end += real_shift

        # Validate length if binning
        if self.bin_size > 1:
            assert start % self.bin_size == 0, f"Post-shift start {start} must be divisible by bin_size {self.bin_size} if binning data"
            assert end % self.bin_size == 0, f"Post-shift end {end} must be divisible by bin_size {self.bin_size} if binning data"

        # If pre-binned, divide by bin size to get bin coordinates rather than bp coordinates
        if self.prebinned:
            start = start//self.bin_size
            end = end//self.bin_size

        # Get values from saved values or by reading in on the fly
        if self.in_memory:
            assert chrom in self.data, f"Could not find chromosome {chrom} in read-in data (chromosomes {list(self.data.keys())})"
            if self.compressed:
                values = self.data[chrom][start:end, :].toarray()
            else:
                values = self.data[chrom][start:end, :]
            if strand == "-":
                values = np.flip(values, axis = 0)
        else:
            values = self._get_single_region(chrom, start, end, strand)

        # Convert to float32 (pybigtools returns float64 by default, while our in_memory bws are in float16 to save memory)
        values = values.astype('float32')

        # Do binning if not pre-binned
        if not self.prebinned and self.bin_size > 1:
            values = values.reshape((-1, self.bin_size, values.shape[-1]))
            if self.target == 'mean':
                values = values.mean(axis=1)
            elif self.target == 'max':
                values = values.max(axis=1)
            elif self.target == 'count' or self.target == 'sum':
                values = values.sum(axis=1)
            elif self.target == 'logcount' or self.target == 'logsum':
                values = np.log1p(values.sum(axis=1))

        return values

    def _read_bws(self):
        """"""
        # Initialize empty arrays per chromosome
        if self.prebinned:
            self.data = {chrom: np.zeros((chrom_size, len(self.paths)), dtype='float16') for chrom, chrom_size in self.binned_chrom_sizes.items()}
        else:
            self.data = {chrom: np.zeros((chrom_size, len(self.paths)), dtype='float16') for chrom, chrom_size in self.chrom_sizes.items()}

        # Read in data per path
        file_objects = {path: pybigtools.open(path, "r") for path in self.paths}

        # TODO maybe: parallelize across chroms as well (i.e. chrom*file parallel entries), but risk of multi-IO
        # TODO maybe: check if other parallelizations work faster
        # Read in one chrom at a time so that we can compress the entire chrom readout before moving to the next one
        try:
            for chrom in tqdm.tqdm(self.chrom_sizes):
                # Calculate bins if reading in pre-binned
                if self.prebinned:
                    end = self.binned_chrom_sizes[chrom] * self.bin_size
                    bins = end // self.bin_size
                else:
                    end = None
                    bins = None
                # Extract values for this chromosome per opened bw
                with ThreadPoolExecutor() as executor:
                    futures_to_idxs = {
                        executor.submit(
                            _get_chrom,
                            bw=file_objects[bw_path],
                            chrom=self.chrom_name_mapping[bw_path][chrom],
                            end=end,
                            bins=bins,
                            target=self.target,
                        ): path_i
                        for path_i, bw_path in enumerate(self.paths)
                    }
                    for future in futures_to_idxs:
                        path_i = futures_to_idxs[future]
                        self.data[chrom][:, path_i] = future.result()

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
            values = np.flip(values, axis=0)
        # TODO: implement binning through pybigtools directly
        return values

    @property
    def shape(self):
        return len(self.class_names), len(self.chrom_sizes)

    @property
    def obs_names(self):
        return self.class_names

    @property
    def n_obs(self):
        return len(self.class_names)

    @property
    def chrom_names(self):
        return list(self.chrom_sizes.keys())

    @property
    def n_chroms(self):
        return len(self.chrom_sizes)

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        """String representation of the trackdata."""
        repr_str = (
            f"TrackData object with {self.n_obs} tracks and {self.n_chroms} chromosomes. \nin_memory: {self.in_memory}, prebinned: {self.prebinned}, bin_size: {self.bin_size}, target: {self.target}\n"
            f"Tracks: {self.obs_names}\n"
            f"Chromosomes: {self.chrom_names}"
        )
        return repr_str


def _get_chrom(bw, chrom, end=None, bins=None, target='mean'):
    if bins is not None:
        if target == 'mean':
            summary = 'mean'
        elif target == 'max':
            summary = 'max'
        elif target == 'count' or target == 'sum':
            raise ValueError("target=count is not supported for now")
        elif target == 'logcount' or target == 'logsum':
            raise ValueError("target=logcount is not supported for now")
    else:
        summary = 'mean' # keep default if not binning, doesn't do anything anyway
    values = bw.values(
        chrom,
        end=end,
        bins=bins,
        summary=summary,
        exact=True,
        missing=0.0,
        oob=0.0,
    )
    values = values.astype('float16')
    if target == 'logcount' or target == 'logsum':
        values = np.log1p(values)
    return values
