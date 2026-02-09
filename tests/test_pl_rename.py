"""Test the old plotting function names, while we still support them."""
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest

import crested.pl

from ._utils import create_anndata_with_regions, generate_simulated_patterns


# ----------- Test bar -----------
def test_bar_normalization_weights():
    regions = [f"chr{chr_i}:{start}-{start+100}" for chr_i in range(10) for start in range(100, 2000, 100)]
    adata = create_anndata_with_regions(regions, random_state=42)
    crested.pp.normalize_peaks(adata, gini_std_threshold = 0.1, top_k_percent=0.5)
    fig, ax = crested.pl.bar.normalization_weights(
        adata=adata,
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

def test_bar_region(adata_preds):
    fig, ax = crested.pl.bar.region(
        adata_preds,
        region="chr1:194208032-194208532",
        target='model_1',
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

def test_bar_region_predictions(adata_preds):
    # Default function
    fig, axs = crested.pl.bar.region_predictions(
        adata=adata_preds,
        region="chr1:92202766-92203266",
        show=False
    )
    assert len(axs) == 3
    assert fig is not None and axs is not None
    plt.close()

def test_bar_predictions():
    prediction = np.abs(np.random.randn(19))
    classes = [f"class_{i}" for i in range(19)]
    fig, ax = crested.pl.bar.prediction(
        prediction=prediction,
        classes=classes,
        plot_kws={'alpha': 0.5},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

# ----------- Test heatmap -----------
def test_heatmap_self(adata_preds):
    # Default function
    fig, ax = crested.pl.heatmap.correlations_self(
        adata=adata_preds,
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

def test_heatmap_pred(adata_preds):
    # Plot single model with default-ish plot
    fig, ax = crested.pl.heatmap.correlations_predictions(
        adata=adata_preds,
        model_names='model_1',
        split='test',
        cbar=False,
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

# ----------- Test hist -----------
def test_hist_distribution(adata_preds):
    # Test simple plot
    fig, axs = crested.pl.hist.distribution(
        adata=adata_preds,
        split="test",
        show=False
    )
    assert len(axs) == 12
    assert fig is not None and axs is not None
    plt.close()

# ----------- Test scatter -----------
def test_scatter_class_density(adata_preds):
    # Plot single class for two models without density
    fig, ax = crested.pl.scatter.class_density(
        adata=adata_preds,
        split=None,
        class_name=adata_preds.obs_names[1],
        log_transform=False,
        exclude_zeros=False,
        density_indication=False,
        square=False,
        identity_line=False,
        cbar=False,
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

# ----------- Test violin -----------
def test_violin_correlations(adata_preds):
    fig, ax = crested.pl.violin.correlations(
        adata=adata_preds,
        split=None,
        log_transform=True,
        plot_kws={'saturation': 0.5},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

# ----------- Test patterns -----------
def test_patterns_contribution_scores():
    scores = np.random.uniform(-1, 3, (1, 1, 100, 4))
    seqs_one_hot = np.eye(4)[None, np.random.randint(4, size=100)]
    # Simple plot
    fig, ax = crested.pl.patterns.contribution_scores(
        scores, seqs_one_hot, show=False
    )
    assert fig is not None and ax is not None
    # Extensive plot
    fig, ax = crested.pl.patterns.contribution_scores(
        scores,
        seqs_one_hot,
        "chr1:100-200",
        "celltype_A",
        zoom_n_bases=50,
        highlight_positions=(50, 60),
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

def test_patterns_contribution_scores_mutagenesis():
    scores = np.random.uniform(-3, 1, (1, 1, 100, 4))
    seqs_one_hot = np.eye(4)[None, np.random.randint(4, size=100)]
    # Plot mutagenesis scatter
    fig, ax = crested.pl.patterns.contribution_scores(
        scores,
        seqs_one_hot,
        "chr1:100-200",
        "celltype_A",
        zoom_n_bases=50,
        highlight_positions=(50, 60),
        method="mutagenesis",
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

    # Plot mutagenesis letters
    fig, ax = crested.pl.patterns.contribution_scores(
        scores,
        seqs_one_hot,
        "chr1:100-200",
        "celltype_A",
        zoom_n_bases=50,
        highlight_positions=(50, 60),
        method="mutagenesis_letters",
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

@pytest.fixture(scope="module")
def all_patterns():
    return generate_simulated_patterns()


@pytest.fixture(scope="module")
def all_classes():
    return [
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

@pytest.fixture(scope="module")
def save_dir():
    path = "tests/data/pl_output"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def test_patterns_selected_instances(all_patterns, save_dir):
    pattern_indices = [0, 1]
    crested.pl.patterns.selected_instances(
        pattern_dict=all_patterns,
        idcs=pattern_indices,
    )
    plt.close()


def test_patterns_class_instances(all_patterns, save_dir):
    crested.pl.patterns.class_instances(
        all_patterns, idx=2, class_representative=True
    )
    plt.close()


def test_patterns_clustermap(all_patterns, all_classes, save_dir):
    pytest.importorskip("modiscolite")
    pattern_matrix = crested.tl.modisco.create_pattern_matrix(
        classes=all_classes, all_patterns=all_patterns, normalize=True
    )
    pat_seqs = crested.tl.modisco.generate_nucleotide_sequences(all_patterns)
    crested.pl.patterns.clustermap(
        pattern_matrix,
        classes=all_classes,
        subset=["Astro", "OPC", "Oligo"],
        pat_seqs=pat_seqs,
        grid=True,
        height=2,
        width=20,
    )
    plt.close()

def test_patterns_similarity_heatmap(all_patterns, save_dir):
    pytest.importorskip("modiscolite")
    pytest.importorskip("memelite")
    sim_matrix, indices = crested.tl.modisco.calculate_similarity_matrix(all_patterns)
    crested.pl.patterns.similarity_heatmap(
        sim_matrix, indices=indices
    )
    plt.close()

if __name__ == "__main__":
    pytest.main()
