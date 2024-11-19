import pytest

import crested

from ._utils import create_anndata_with_regions


def test_train_val_test_split_by_region():
    regions = [
        "chr1:100-200",
        "chr1:300-400",
        "chr2:100-200",
        "chr2:300-400",
        "chr3:100-200",
    ]
    adata = create_anndata_with_regions(regions)

    crested.pp.train_val_test_split(
        adata,
        strategy="region",
        val_size=0.2,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    split_counts = adata.var["split"].value_counts()
    assert split_counts["val"] == 1
    assert split_counts["test"] == 1
    assert split_counts["train"] == 3


def test_train_val_test_split_by_chromosome():
    regions = [
        "chr1:100-200",
        "chr1:300-400",
        "chr2:100-200",
        "chr2:300-400",
        "chr3:100-200",
    ]
    adata = create_anndata_with_regions(regions)

    crested.pp.train_val_test_split(
        adata,
        strategy="chr",
        val_chroms=["chr1"],
        test_chroms=["chr2"],
    )

    split_counts = adata.var["split"].value_counts()
    assert split_counts["val"] == 2
    assert split_counts["test"] == 2
    assert split_counts["train"] == 1


def test_train_val_test_split_by_chromosome_auto():
    regions = [
        "chr1:100-200",
        "chr2:100-200",
        "chr3:300-400",
        "chr4:100-200",
        "chr5:100-200",
        "chr6:100-200",
        "chr7:100-200",
        "chr8:100-200",
        "chr9:100-200",
        "chr10:100-200",
    ]
    adata = create_anndata_with_regions(regions)

    crested.pp.train_val_test_split(
        adata,
        strategy="chr_auto",
        val_size=0.2,
        test_size=0.2,
        random_state=None,
    )

    val_count = adata.var["split"].value_counts().get("val", 0)
    test_count = adata.var["split"].value_counts().get("test", 0)
    train_count = adata.var["split"].value_counts().get("train", 0)

    total_count = len(regions)
    val_fraction = 0.2
    test_fraction = 0.2
    train_fraction = 1 - val_fraction - test_fraction

    assert val_count / total_count == pytest.approx(val_fraction, rel=1e-2)
    assert test_count / total_count == pytest.approx(test_fraction, rel=1e-2)
    assert train_count / total_count == pytest.approx(train_fraction, rel=1e-2)


# def test_normalization_consistency():
#     regions = [
#         "chr1:100-200",
#         "chr1:300-400",
#         "chr2:100-200",
#         "chr2:300-400",
#         "chr3:100-200",
#     ]
#     adata_dense = create_anndata_with_regions(regions, random_state=42)
#     adata_sparse = create_anndata_with_regions(regions, compress=True, random_state=42)

#     normalized_dense = crested.pp.normalize_peaks(
#         adata_dense,
#         peak_threshold=0.2,
#         gini_std_threshold=1.0,
#         top_k_percent=0.2,
#     )
#     normalized_sparse = crested.pp.normalize_peaks(
#         adata_sparse,
#         peak_threshold=0.2,
#         gini_std_threshold=1.0,
#         top_k_percent=0.2,
#     )

#     normalized_sparse_dense = normalized_sparse.X.toarray()

#     # Check that both normalized datasets are identical
#     np.testing.assert_array_almost_equal(
#         normalized_dense.X,
#         normalized_sparse_dense,
#         decimal=5,
#         err_msg="Normalized results differ between dense and sparse formats.",
#     )
