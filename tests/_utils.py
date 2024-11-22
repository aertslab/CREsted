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
    if random_state is not None:
        np.random.seed(random_state)
    data = np.abs(np.random.randn(n_classes, len(regions)))
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
