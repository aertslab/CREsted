import pytest
import numpy as np
from numpy import array_equiv

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

def test_train_val_test_split_inplace():
    regions = [
        "chr1:100-200",
        "chr1:300-400",
        "chr2:100-200",
        "chr2:300-400",
        "chr3:100-200",
    ]

    adata = create_anndata_with_regions(regions)
    adata_inplace = adata.copy()

    adata_copy = crested.pp.train_val_test_split(
        adata,
        strategy="chr",
        val_chroms=["chr1"],
        test_chroms=["chr2"],
        inplace=False
    )
    crested.pp.train_val_test_split(
        adata_inplace,
        strategy="chr",
        val_chroms=["chr1"],
        test_chroms=["chr2"],
        inplace=True
    )

    assert adata_inplace.var.equals(adata_copy.var)
    assert not adata.var.equals(adata_inplace.var)
    assert not adata.var.equals(adata_copy.var)

def test_filter_regions_on_specificity_inplace(adata_function):
    adata_inplace = adata_function.copy()

    adata_copy = crested.pp.filter_regions_on_specificity(adata_function, inplace=False)
    crested.pp.filter_regions_on_specificity(adata_inplace, inplace=True)
    assert array_equiv(adata_inplace.X, adata_copy.X)
    assert not adata_function.X == pytest.approx(adata_inplace.X)
    assert not adata_function.X == pytest.approx(adata_copy.X)

def test_sort_and_filter_regions_on_specificity_inplace(adata_function):
    adata_inplace = adata_function.copy()

    adata_copy = crested.pp.sort_and_filter_regions_on_specificity(adata_function, top_k=3, inplace=False)
    crested.pp.sort_and_filter_regions_on_specificity(adata_inplace, top_k=3, inplace=True)
    assert array_equiv(adata_inplace.X, adata_copy.X)
    assert not adata_function.X == pytest.approx(adata_inplace.X)
    assert not adata_function.X == pytest.approx(adata_copy.X)

def test_normalize_peaks_inplace():
    # Create larger anndata to prevent zero division issues from presumably normalizing on 0/1 peak
    adata = create_anndata_with_regions([f'chr{chr_i}:{start}-{start+100}' for chr_i in range(1, 10) for start in range(0, 1000, 100)])
    adata_inplace = adata.copy()

    adata_copy, _ = crested.pp.normalize_peaks(adata, peak_threshold=0.2, gini_std_threshold=0, top_k_percent=0.2, inplace=False)
    _ = crested.pp.normalize_peaks(adata_inplace, peak_threshold=0.2, gini_std_threshold=0, top_k_percent=0.2, inplace=True)
    assert adata_inplace.X == pytest.approx(adata_copy.X)
    assert not adata.X == pytest.approx(adata_inplace.X)
    assert not adata.X == pytest.approx(adata_copy.X)

def test_normalize_peaks_uses_the_cell_types_own_top_peaks():
    """Gini scores must come from the peaks the top-k selection actually picked.

    Regions at or below `peak_threshold` are dropped before sorting, which offsets
    sort positions from region indices by the number of regions dropped ahead of
    them, so positions have to be mapped back before indexing the matrix.

    The fixture has a closed-form answer: 100 broad regions open in every cell type
    at that cell type's own height, plus 20 per cell type that are tall in one cell
    type and zero elsewhere. Broad regions get a low Gini and specific ones a high
    Gini, so each cell type's low-Gini subset is exactly its broad regions and its
    top_k_mean is exactly its broad height.
    """
    heights = np.array([1.0, 2.0, 4.0, 8.0])
    broad = np.tile(heights, (100, 1))
    specific = np.zeros((20 * len(heights), len(heights)))
    for c in range(len(heights)):
        specific[c * 20 : (c + 1) * 20, c] = 100.0

    matrix = np.vstack([broad, specific])
    matrix = matrix[np.random.default_rng(0).permutation(len(matrix))]  # index != sort position

    regions = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(len(matrix))]
    adata = create_anndata_with_regions(regions, n_classes=len(heights))
    adata.X = matrix.T.copy()  # AnnData is (cell types, regions)

    adata_out, _ = crested.pp.normalize_peaks(
        adata, peak_threshold=0.0, gini_std_threshold=0, top_k_percent=1.0, inplace=False
    )

    weights = np.asarray(adata_out.obsm["weights"]).ravel()
    assert weights == pytest.approx(heights.max() / heights)


@pytest.mark.parametrize("case", ["no_broad_peaks", "all_below_threshold", "top_k_rounds_to_zero"])
def test_normalize_peaks_raises_when_no_peaks_are_selected(case):
    """A cell type with no selected peaks has an undefined weight, so raise.

    Three routes there: no top peak counts as broad (raising `gini_std_threshold`
    lowers the cutoff past some dataset dependent point), every region at or below
    `peak_threshold`, or so few regions above it that `top_k_percent` rounds the
    selection to zero. The latter two leave `top_indices` empty, which is a clean
    shape-(0,) reduction over the cell type axis rather than an error, so they reach
    this check as well. Dividing by the resulting zero would otherwise put inf (or
    nan, if it happens to every cell type) into .X.
    """
    adata = create_anndata_with_regions(
        [f"chr{chr_i}:{start}-{start + 100}" for chr_i in range(1, 10) for start in range(0, 1000, 100)],
        random_state=0,
    )
    kwargs = {"gini_std_threshold": 1.0, "top_k_percent": 0.2}

    if case == "no_broad_peaks":
        kwargs["gini_std_threshold"] = 10
    elif case == "all_below_threshold":
        adata.X[0] = 0.0
    else:
        adata.X[0] = 0.0
        adata.X[0, :2] = 3.0
        kwargs["top_k_percent"] = 0.1  # int(2 * 0.1) == 0 -> nothing selected

    with pytest.raises(ValueError, match="normalization weight is undefined"):
        crested.pp.normalize_peaks(adata, peak_threshold=0.0, inplace=False, **kwargs)


def test_change_regions_width_inplace(adata_function):
    adata_inplace = adata_function.copy()

    adata_copy = crested.pp.change_regions_width(adata_function, width=888, inplace=False)
    crested.pp.change_regions_width(adata_inplace, width=888, inplace=True)

    assert adata_inplace.var.equals(adata_copy.var)
    assert not adata_function.var.equals(adata_inplace.var)
    assert not adata_function.var.equals(adata_copy.var)

def test_change_regions_width(adata):
    adata_resized = crested.pp.change_regions_width(adata, width=888, inplace=False)
    assert adata_resized.var_names[0] == crested.utils.resize_region(adata.var_names[0], 888)
    assert adata_resized.var_names[0] == "chr1:194207838-194208726"

def test_change_regions_width_drops_resized_out_of_bounds():
    """Regions whose *resized* window falls off a contig edge must be dropped.

    The original peaks are all within the chromosome, but after widening, the
    ones near a contig edge run past 0 or past the chromosome length. These must
    be removed (the boundary check has to use the resized coordinates, not the
    original ones). chrM in test.chrom.sizes is 16299 bp.
    """
    regions = [
        "chrM:50-150",  # center 100 -> [-957, 1157]: negative start -> drop
        "chr1:100000-100100",  # center 100050 -> [98993, 101107]: in bounds -> keep
        "chrM:16200-16280",  # center 16240 -> [15183, 17297]: past end 16299 -> drop
    ]
    adata = create_anndata_with_regions(regions)

    crested.pp.change_regions_width(
        adata, width=2114, chromsizes_file="tests/data/test.chrom.sizes"
    )

    assert list(adata.var_names) == ["chr1:98993-101107"]
    assert adata.n_vars == 1


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
