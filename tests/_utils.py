"""Utils for testing functions."""

import random

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def create_anndata_with_regions(
    regions: list[str],
    n_classes: int = 10,
    chr_var_key: str = "chr",
    compress: bool = False,
    random_state: int = None,
) -> ad.AnnData:
    """Create an AnnData object with given regions."""
    rng = np.random.default_rng(random_state)
    data = np.abs(rng.standard_normal((n_classes, len(regions))))
    var = pd.DataFrame(index=regions)
    var[chr_var_key] = [region.split(":")[0] for region in regions]
    var["start"] = [int(region.split(":")[1].split("-")[0]) for region in regions]
    var["end"] = [int(region.split(":")[1].split("-")[1]) for region in regions]

    if compress:
        data = sp.csr_matrix(data)

    anndata = ad.AnnData(X=data, var=var)
    anndata.obs_names = [f"cell_{i}" for i in range(n_classes)]
    anndata.var_names = regions
    return anndata

def generate_simulated_patterns(
    num_patterns=10,
    seqlet_length=50,
    num_instances=20,
    seed=42,
    cell_classes: list[str] = None,
):
    """Generate 'all_patterns' dict to test pattern plotting funcitons."""
    np.random.seed(seed)
    random.seed(seed)
    if cell_classes is None:
        cell_classes = [
            "Astro",
            "Endo",
            "L2_3IT",
            "L5ET",
            "L5IT",
            "L5_6NP",
            "L6CT",
            "L6IT",
            "L6b",
            "Micro_PVM",
            "Oligo",
            "Pvalb",
            "Sst",
            "SstChodl",
            "VLMC",
            "Lamp5",
            "OPC",
            "Sncg",
            "Vip",
        ]

    simulated_data = {}
    for i in range(num_patterns):
        # Generate instances
        instances = {}
        for _ in range(num_instances):
            cell_class = random.choice(cell_classes)
            pattern_type = random.choice(["neg", "pos"])
            pattern_index = random.randint(0, 30)
            instance_key = f"{cell_class}_{pattern_type}_patterns_{pattern_index}"

            instances[instance_key] = {
                "sequence": np.random.dirichlet(np.ones(4), size=4),
                "contrib_scores": np.random.randn(4, 4) * 0.01,
                "hypothetical_contribs": np.random.randn(4, 4) * 0.01,
                "seqlets": {
                    "contrib_scores": np.random.randn(seqlet_length, 4, 4) * 0.01,
                    "sequence": np.random.dirichlet(
                        np.ones(4), size=(seqlet_length, 4)
                    ),
                    "class": random.choice(["Vip", "NonVip"]),
                },
            }

        # Add data to the main pattern
        pattern_type = random.choice(["neg", "pos"])
        pattern_id = f"{random.choice(cell_classes)}_{pattern_type}_patterns_{random.randint(0, 30)}"

        simulated_data[str(i)] = {
            "pattern": {
                "sequence": np.random.dirichlet(np.ones(4), size=4),
                "contrib_scores": np.random.randn(4, 4) * 0.01,
                "hypothetical_contribs": np.random.randn(4, 4) * 0.01,
                "seqlets": {
                    "contrib_scores": np.random.randn(seqlet_length, 4, 4) * 0.01,
                    "sequence": np.random.dirichlet(
                        np.ones(4), size=(seqlet_length, 4)
                    ),
                    "class": random.choice(["Vip", "NonVip"]),
                },
                "id": pattern_id,
                "pos_pattern": pattern_type == "pos",
                "ic": np.float32(np.random.uniform(0.5, 2.0)),  # Simulating ic values
                "ppm": np.random.dirichlet(np.ones(4), size=4).astype(np.float32),
                "class": pattern_id.split("_")[0],
            },
            "ic": np.random.rand(),
            "classes": {
                class_name: {
                    "sequence": np.random.dirichlet(np.ones(4), size=4),
                    "contrib_scores": np.random.randn(4, 4) * 0.01,
                    "hypothetical_contribs": np.random.randn(4, 4) * 0.01,
                    "seqlets": {
                        "contrib_scores": np.random.randn(seqlet_length, 4, 4) * 0.01,
                        "sequence": np.random.dirichlet(
                            np.ones(4), size=(seqlet_length, 4)
                        ),
                        "class": random.choice(["Vip", "NonVip"]),
                    },
                    "id": f"{class_name}_patterns_{random.randint(0, 30)}",
                    "pos_pattern": random.choice([True, False]),
                    "ic": np.float32(np.random.uniform(0.5, 2.0)),
                    "ppm": np.random.dirichlet(np.ones(4), size=4).astype(np.float32),
                    "class": class_name,
                    "n_seqlets": random.randint(1, 10),
                }
                for class_name in cell_classes
            },
            "instances": instances,
        }
    return simulated_data


def _seqs_to_onehot(seqs: list[str]) -> np.ndarray:
    """Stack equal-length ACGT strings into a (n_seqs, length, 4) one-hot array."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    length = len(seqs[0])
    onehot = np.zeros((len(seqs), length, 4))
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            onehot[i, j, mapping[ch]] = 1.0
    return onehot


def make_trimmed_pattern(seqs: list[str], n_seqlets: int) -> dict:
    """Build one 'trimmed pattern' dict in the shape `_read_and_trim_patterns` returns.

    `seqs` are equal-length ACGT strings whose per-column frequencies define the PPM
    (so a single repeated consensus -> high IC, a mixture -> lower IC); `n_seqlets`
    is the reported seqlet support that drives representative/sort selection.
    `_process_patterns_agglomerative` recomputes ppm/ic/n_seqlets from this dict.
    """
    return {
        "seqlets": {
            "sequence": _seqs_to_onehot(seqs),
            "n_seqlets": np.array([n_seqlets]),
        }
    }


def synthetic_matched_patterns():
    """Synthetic per-class patterns for exercising `process_patterns` clustering.

    Returns ``(matched_files, fake_read)`` where `fake_read` is a drop-in for
    ``crested.tl.modisco._tfmodisco._read_and_trim_patterns``: it ignores the file
    path and returns a fixed ``(patterns, ids, is_pos)`` per cell type, so the
    clustering path can run without real modisco HDF5 files. IDs are deterministic
    (``"<class>_p<i>"``) and independent of dict order.
    """
    # Two distinct high-IC consensus motifs (repeated so the consensus dominates the
    # pseudocount -> sharp PPM), and a near-uniform mixture -> low-IC motif.
    sharp1 = ["ACGTACGTACGT"] * 20
    sharp2 = ["TGCATGCATGCA"] * 20
    fuzzy = ["AAAACCCCGGGG", "TTTTGGGGCCCC", "CCCCAAAATTTT", "GGGGTTTTAAAA"] * 5
    per_class = {
        # Most-supported member overall is fuzzy (LOW IC) -> becomes the
        # representative, but is NOT the highest-IC member, so cluster IC (max over
        # members) must differ from the representative's IC.
        "classA": [
            (make_trimmed_pattern(fuzzy, 30), True),
            (make_trimmed_pattern(fuzzy, 10), True),  # same motif, fewer seqlets
        ],
        "classB": [
            (make_trimmed_pattern(sharp1, 20), True),
            (make_trimmed_pattern(sharp2, 15), True),
        ],
        "classC": [
            (make_trimmed_pattern(sharp1, 5), False),
            (make_trimmed_pattern(sharp2, 8), False),
        ],
    }

    def fake_read(cell_type, file_list, trim_ic_threshold, verbose):
        items = per_class[cell_type]
        patterns = [p for p, _ in items]
        ids = [f"{cell_type}_p{i}" for i in range(len(items))]
        is_pos = [pos for _, pos in items]
        return patterns, ids, is_pos

    matched_files = {ct: f"{ct}.h5" for ct in per_class}
    return matched_files, fake_read
