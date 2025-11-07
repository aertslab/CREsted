import os
from collections.abc import Sequence
from copy import copy

import numpy as np
import pybigtools
import tqdm
from loguru import logger
from scipy.sparse import csc_array

from crested._io import _extract_tracks_from_bigwig


class TrackData:
    def __init__(
        self,
        paths: dict[str] | list[str] | dict[tuple(str, str)],
        contigs:  list[str] | list[str, int, int] | None = None, #TODO: think about what to do with regions
        bin_size: int | None = None,
        target: str = "mean",
        in_memory: bool = True,
        chrom_mapping: dict | None = None,
    ):
        """
        
        
        contigs
            Optional: list of regions to read in. 
        """
        # Check and save paths and path_names
        if isinstance(paths, Sequence):
            if not all(isinstance(path, str) for path in paths):
                raise ValueError("If providing a list of paths, you can only enter one path per dataset."
                "To use stranded bigwigs, please submit paths as a dict like {ct1: (ct1_pos_path.bw, ct1_neg_path.bw)} instead.}")
            self.paths = paths
            self.path_names = [os.path.splitext(os.path.basename(path))[0] for path in paths]
        else:
            self.paths = list(paths.values())
            self.path_names = list(paths.keys())

        # Check if any of them are stranded pairs
        self.stranded = [not isinstance(path, str) for path in self.paths]

        # Save other arguments
        self.contigs = contigs
        self.bin_size = bin_size
        self.target = target
        self.in_memory = in_memory

        # Check whether paths truly exist
        logger.info("Checking whether bigwigs exist")
        for path, stranded_status in zip(self.paths, self.stranded):
            if stranded_status:
                assert pybigtools.open(path[0]).is_bigwig, f"{path[0]} does not appear to be a bigwig file."
                assert pybigtools.open(path[1]).is_bigwig, f"{path[1]} does not appear to be a bigwig file."
            else:
                assert pybigtools.open(path).is_bigwig, f"{path} does not appear to be a bigwig file."
        with pybigtools.open(self.paths[0]) as f:
            self.chromsizes = copy(f.chroms())
        # self.chrom_mapping = ?

        if self.in_memory:
            logger.info("Reading bigwigs into memory")
            self._read_bws()

        # TODO maybe: add support for stranded bigwigs (provide pair of paths, save both in alternative version, but bit of a bother to implement)

    def __getitem__(self, idx):
        """"""
        # Parse string index
        if isinstance(idx, str):
            n_colons = idx.count(':')
            if n_colons == 2:
                chrom, start_end, strand = idx.split(':')
            elif n_colons == 1:
                chrom, start_end = idx.split(':')
                strand = "+"
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

        # TODO: add chr prefix fuzzyness

        # Get values from saved values or by reading in on the fly
        if self.in_memory:
            assert chrom in self.data, f"Could not find chromosome {chrom} in read-in data (chromosomes {list(self.data.keys())})"
            return self.data[chrom][:, start:end].toarray()
        else:
            return self._get_single_region(chrom, start, end, strand)

    def _read_bws(self):
        """"""
        # Initialize empty arrays per chromosome
        self.data = {chrom_name: np.zeros((len(self.paths), chrom_size), dtype='float64') for chrom_name, chrom_size in self.chromsizes.items()}
        if any(self.stranded):
            self.data_negative = {chrom_name: if stranded else NotImplementedError}

        # Read in data per path
        for path_i, bw_path in enumerate(tqdm.tqdm(self.paths)):
            # Read in all chromosomes of this file
            # TODO: rewrite to keep file open but support different length readouts
            with pybigtools.open(bw_path, "r") as bw:
                    results = []
                    for chrom in self.chromsizes.keys():
                        arr = np.empty(
                            binned_length, dtype="float64"
                        )  # pybigtools returns values in float64
                        chrom, start, end = region

                        # Extract values
                        results.append(
                            bw.values(
                                chrom,
                                bins=bins,
                                summary=target,
                                exact=exact,
                                missing=missing,
                                oob=oob,
                                arr=self.data[chrom_name][path_i, ...],
                            )
                        )
            # for chrom_i, (chrom_name, chrom_size) in enumerate(self.chromsizes.items()):
                # values = _extract_tracks_from_bigwig(
                #     bw_file = bw_path,
                #     coordinates = [coordinates[chrom_i]],
                #     # bin_size = self.bin_size, # do later instead when gathering sequences
                #     target=self.target
                # )
                # # Save values to merged dataset
                # self.data[chrom_name][path_i, ...] = values[0, ...].astype('float32')

        # Convert values to column sparse arrays
        for chrom in self.data:
            self.data[chrom] = csc_array(self.data[chrom])

    def _get_single_region(self, chrom, start, end, strand = "+"):
        """"""
        values = np.vstack([
            _extract_tracks_from_bigwig(
                bw_file = bw_path,
                coordinates = [[chrom, start, end]],
                target=self.target
            ) for bw_path in self.paths
        ])
        if strand == "-":
            values = np.flip(values, axis=-1) # TODO: test
        return values

    @property
    def obs_names(self):
        return self.path_names

    def __len__(self):
        return len(self.paths)
