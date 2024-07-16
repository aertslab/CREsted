"""Utils for testing functions."""

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
    """
    Utility function to create an AnnData object with given regions, with options for compression
    and reproducibility.
    """
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
