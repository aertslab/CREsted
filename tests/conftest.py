"""Fixtures to be used by all unit tests."""

import keras
import numpy as np
import pytest

import crested
import crested._conf as conf
from crested._genome import Genome

from ._utils import create_anndata_with_regions

np.random.seed(42)
keras.utils.set_random_seed(42)


class MockFastaFile:
    """Mock pysam FastaFile that generates random sequences on-the-fly."""

    def __init__(self, seed=42):
        """Initialize with a seed for reproducible random sequences."""
        self.seed = seed
        # Use chromsizes from test.chrom.sizes to match test data expectations
        self.references = [
            "chr1",
            "chr10",
            "chr11",
            "chr12",
            "chr13",
            "chr14",
            "chr15",
            "chr16",
            "chr17",
            "chr18",
            "chr19",
            "chr2",
            "chr3",
            "chr4",
            "chr5",
            "chr6",
            "chr7",
            "chr8",
            "chr9",
            "chrM",
            "chrX",
            "chrY",
        ]
        self.lengths = [
            195471971,
            130694993,
            122082543,
            120129022,
            120421639,
            124902244,
            104043685,
            98207768,
            94987271,
            90702639,
            60790335,
            182113224,
            160039680,
            156508116,
            151834684,
            149736546,
            145441459,
            129401213,
            124595110,
            16299,
            171031299,
            91744698,
        ]
        self.filename = b"mock_genome.fa"

    def fetch(self, reference: str, start: int, end: int) -> str:
        """Generate a reproducible random DNA sequence for the given coordinates."""
        # Use coordinates as seed modifier for reproducibility
        rng = np.random.RandomState(self.seed + hash((reference, start, end)) % (2**31))
        length = end - start
        bases = ["A", "C", "G", "T"]
        return "".join(rng.choice(bases, size=length))

    def close(self):
        """Mock close method."""
        pass


class MockGenome(Genome):
    """Mock Genome that generates random sequences instead of reading from file."""

    def __init__(self, name="mock_genome", seed=42):
        """Initialize mock genome, bypassing file checks."""
        # Skip parent __init__ to avoid file existence checks
        self._name = name
        self._chrom_sizes = None  # Will be inferred from mock fasta
        self._annotation = None
        self._seed = seed
        self._mock_fasta = MockFastaFile(seed=seed)

    @property
    def fasta(self):
        """Return mock FASTA file instead of loading from disk."""
        return self._mock_fasta


@pytest.fixture(autouse=True)
def reset_genome():
    """Reset global genome state before each test to ensure test isolation."""
    conf.genome = None
    yield
    conf.genome = None


@pytest.fixture(scope="module")
def keras_model():
    """Keras model fixture."""
    from crested.tl.zoo import simple_convnet

    model = simple_convnet(
        seq_len=500,
        num_classes=5,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    return model


@pytest.fixture(scope="module")
def adata():
    """Anndata fixture."""
    regions = [
        "chr1:194208032-194208532",
        "chr1:92202766-92203266",
        "chr1:92298990-92299490",
        "chr1:3406052-3406552",
        "chr1:183669567-183670067",
        "chr1:109912183-109912683",
        "chr1:92210697-92211197",
        "chr1:59100954-59101454",
        "chr1:84634055-84634555",
        "chr1:48792527-48793027",
    ]
    return create_anndata_with_regions(regions)


@pytest.fixture(scope="module")
def genome():
    """Mock genome fixture that generates random sequences."""
    return MockGenome(name="hg38", seed=42)


@pytest.fixture(scope="module")
def adata_specific():
    """Specific anndata fixture."""
    ann_data = crested.import_bigwigs(
        bigwigs_folder="tests/data/test_bigwigs",
        regions_file="tests/data/test_bigwigs/consensus_peaks_subset.bed",
    )
    crested.pp.sort_and_filter_regions_on_specificity(ann_data, top_k=3)
    return ann_data
