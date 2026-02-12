import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest

import crested.pl

from ._utils import create_anndata_with_regions, generate_simulated_patterns


# ----------- Test create_plot -----------
def test_create_plot_new():
    fig, ax = crested.pl.create_plot(
        ax=None,
        kwargs_dict={}
    )
    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    plt.close()

def test_create_plot_existing():
    fig, ax = plt.subplots()
    fig_create, ax_create = crested.pl.create_plot(
        ax=ax,
        kwargs_dict={}
    )
    assert fig is fig_create and ax is ax_create
    plt.close()

def test_create_plot_multiple():
    fig, ax = crested.pl.create_plot(ax=None, kwargs_dict={}, nrows=2, ncols=3)
    assert fig is not None and ax.shape == (2, 3)
    plt.close()

def test_create_plot_input_check():
    with pytest.raises(ValueError):
        fig, axs = plt.subplots(2)
        crested.pl.create_plot(ax=axs, kwargs_dict={})

def test_create_plot_size_default():
    fig, ax = crested.pl.create_plot(ax=None, kwargs_dict={}, default_width=10, default_height=11)
    assert fig.get_figwidth() == 10. and fig.get_figheight() == 11.
    plt.close()

def test_create_plot_size_kwargs():
    fig, ax = crested.pl.create_plot(ax=None, kwargs_dict={'width': 10, 'height': 11})
    assert fig.get_figwidth() == 10. and fig.get_figheight() == 11.
    plt.close()

def test_create_plot_shareax_default():
    # Share x-axis - default parameters
    fig, axs = crested.pl.create_plot(ax=None, kwargs_dict={}, nrows=2)
    assert len(axs[0].get_shared_x_axes().get_siblings(axs[0])) == 1
    # Share x-axis - set as True in default
    fig, axs = crested.pl.create_plot(ax=None, kwargs_dict={}, default_sharex=True, nrows=2)
    assert len(axs[0].get_shared_x_axes().get_siblings(axs[0])) == 2
    plt.close()

    # Share y-axis - default
    fig, axs = crested.pl.create_plot(ax=None, kwargs_dict={}, ncols=2)
    assert len(axs[0].get_shared_y_axes().get_siblings(axs[0])) == 1
    # Share y-axis - set as True in default
    fig, axs = crested.pl.create_plot(ax=None, kwargs_dict={}, default_sharey=True, ncols=2)
    assert len(axs[0].get_shared_y_axes().get_siblings(axs[0])) == 2
    plt.close()

def test_create_plot_shareax_kwargs():
    # Share x-axis - share using kwargs
    fig, axs = crested.pl.create_plot(ax=None, kwargs_dict={'sharex': True}, nrows=2)
    assert len(axs[0].get_shared_x_axes().get_siblings(axs[0])) == 2
    # Share x-axis - override default_sharex with kwargs
    fig, axs = crested.pl.create_plot(ax=None, kwargs_dict={'sharex': False}, default_sharex=True, nrows=2)
    assert len(axs[0].get_shared_x_axes().get_siblings(axs[0])) == 1

    # Share y-axis - share using kwargs
    fig, axs = crested.pl.create_plot(ax=None, kwargs_dict={'sharey': True}, ncols=2)
    assert len(axs[0].get_shared_y_axes().get_siblings(axs[0])) == 2
    # Share y-axis - override default_sharey with kwargs
    fig, axs = crested.pl.create_plot(ax=None, kwargs_dict={'sharey': False}, default_sharey=True, ncols=2)
    assert len(axs[0].get_shared_y_axes().get_siblings(axs[0])) == 1
    plt.close()

# ----------- Test render_plot -----------
def test_render_plot():
    fig, ax = plt.subplots()
    ax.scatter(np.arange(5), np.arange(1, 6))
    fig_render, ax_render = crested.pl.render_plot(fig=fig, axs=ax, show=False)
    assert fig_render is not None and ax_render is not None
    plt.close()

def test_render_plot_title():
    # Test one plot, one title
    fig, ax = plt.subplots()
    fig, ax = crested.pl.render_plot(fig=fig, axs=ax, title='test', title_fontsize=17, show=False)
    assert ax.get_title() == 'test'
    assert ax.title.get_fontsize() == 17.
    plt.close()

    # Test multiple plots, one title
    fig, axs = plt.subplots(2)
    fig, axs = crested.pl.render_plot(fig=fig, axs=axs, title='test', title_fontsize=17, show=False)
    assert all(ax.get_title() == 'test' for ax in axs)
    assert all(ax.title.get_fontsize() == 17. for ax in axs)
    plt.close()

    # Test multiple plots, multiple titles
    fig, axs = plt.subplots(2)
    fig, axs = crested.pl.render_plot(fig=fig, axs=axs, title=['test1', 'test2'], show=False)
    assert [ax.get_title() for ax in axs] == ['test1', 'test2']
    plt.close()

def test_render_plot_suptitle():
    fig, ax = plt.subplots()
    fig, ax = crested.pl.render_plot(fig=fig, axs=ax, suptitle='test', suptitle_fontsize=19, show=False)
    assert fig.get_suptitle() == 'test'
    assert fig._suptitle.get_fontsize() == 19.
    plt.close()

def test_render_plot_label():
    # Test one plot, one label
    fig, ax = plt.subplots()
    fig, ax = crested.pl.render_plot(fig=fig, axs=ax, xlabel='testx', ylabel='testy', xlabel_fontsize=15, ylabel_fontsize=16, show=False)
    assert ax.get_xlabel() == 'testx'
    assert ax.xaxis.label.get_fontsize() == 15.
    assert ax.get_ylabel() == 'testy'
    assert ax.yaxis.label.get_fontsize() == 16.
    plt.close()

    # Test multiple plots, one label
    fig, axs = plt.subplots(2)
    fig, axs = crested.pl.render_plot(fig=fig, axs=axs, xlabel='testx', ylabel='testy', xlabel_fontsize=15, ylabel_fontsize=16, show=False)
    assert all(ax.get_xlabel() == 'testx' for ax in axs)
    assert all(ax.xaxis.label.get_fontsize() == 15. for ax in axs)
    assert all(ax.get_ylabel() == 'testy' for ax in axs)
    assert all(ax.yaxis.label.get_fontsize() == 16. for ax in axs)
    plt.close()

    # Test multiple plots, multiple labels
    fig, axs = plt.subplots(2)
    fig, axs = crested.pl.render_plot(fig=fig, axs=axs, xlabel=['testx1', 'testx2'], ylabel=['testy1', 'testy2'], show=False)
    assert [ax.get_xlabel() for ax in axs] == ['testx1', 'testx2']
    assert [ax.get_ylabel() for ax in axs] == ['testy1', 'testy2']
    plt.close()

def test_render_plot_suplabel():
    fig, ax = plt.subplots()
    fig, ax = crested.pl.render_plot(fig=fig, axs=ax, supxlabel='testx', supylabel='testy', supxlabel_fontsize = 15, supylabel_fontsize = 17, show=False)
    assert fig.get_supxlabel() == 'testx'
    assert fig._supxlabel.get_fontsize() == 15.
    assert fig.get_supylabel() == 'testy'
    assert fig._supylabel.get_fontsize() == 17.
    plt.close()

def test_render_plot_lim():
    # Test one plot, one limit set
    fig, ax = plt.subplots()
    ax.scatter(np.arange(5), np.arange(1, 6))
    fig, ax = crested.pl.render_plot(fig=fig, axs=ax, xlim=(-1, 6), ylim=(0, 7), show=False)
    assert ax.get_xlim() == (-1, 6)
    assert ax.get_ylim() == (0, 7)
    plt.close()

    # Test multiple plots, one limit set
    fig, axs = plt.subplots(2)
    for i in range(len(axs)):
        axs[i].scatter(np.arange(5), np.arange(1, 6))
    fig, axs = crested.pl.render_plot(fig=fig, axs=axs,  xlim=(-1, 6), ylim=(0, 7), show=False)
    assert all(ax.get_xlim() == (-1, 6) for ax in axs)
    assert all(ax.get_ylim() == (0, 7) for ax in axs)
    plt.close()

    # Test multiple plots, multiple limit sets
    fig, axs = plt.subplots(2)
    for i in range(len(axs)):
        axs[i].scatter(np.arange(5), np.arange(1, 6))
    fig, axs = crested.pl.render_plot(fig=fig, axs=axs,  xlim=[(-1, 6), (-0.5, 6.5)], ylim=[(0, 7), (0.5, 7.5)], show=False)
    assert [ax.get_xlim() for ax in axs] == [(-1, 6), (-0.5, 6.5)]
    assert [ax.get_ylim() for ax in axs] == [(0, 7), (0.5, 7.5)]
    plt.close()

def test_render_plot_grid():
    # Test no grid
    fig, ax = plt.subplots()
    ax.scatter(np.arange(5), np.arange(1, 6))
    fig, ax = crested.pl.render_plot(fig=fig, axs=ax, show=False)
    assert not ax.xaxis._major_tick_kw['gridOn'] and not ax.yaxis._major_tick_kw['gridOn']
    plt.close()

    # Test full grid with True alias
    fig, ax = plt.subplots()
    ax.scatter(np.arange(5), np.arange(1, 6))
    fig, ax = crested.pl.render_plot(fig=fig, axs=ax, show=False, grid=True)
    assert ax.xaxis._major_tick_kw['gridOn'] and ax.yaxis._major_tick_kw['gridOn']
    plt.close()

    # Test full grid on multiple plots
    fig, axs = plt.subplots(2)
    for i in range(len(axs)):
        axs[i].scatter(np.arange(5), np.arange(1, 6))
    fig, axs = crested.pl.render_plot(fig=fig, axs=axs, show=False, grid='both')
    assert all(ax.xaxis._major_tick_kw['gridOn'] and ax.yaxis._major_tick_kw['gridOn'] for ax in axs)
    plt.close()

    # Test x grid
    fig, ax = plt.subplots()
    ax.scatter(np.arange(5), np.arange(1, 6))
    fig, ax = crested.pl.render_plot(fig=fig, axs=ax, show=False, grid='x')
    assert ax.xaxis._major_tick_kw['gridOn'] and not ax.yaxis._major_tick_kw['gridOn']
    plt.close()

    # Test y grid
    fig, ax = plt.subplots()
    ax.scatter(np.arange(5), np.arange(1, 6))
    fig, ax = crested.pl.render_plot(fig=fig, axs=ax, show=False, grid='y')
    assert not ax.xaxis._major_tick_kw['gridOn'] and ax.yaxis._major_tick_kw['gridOn']
    plt.close()

def test_render_plot_ticks():
    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, 5), np.arange(1, 6))
    fig, ax = crested.pl.render_plot(fig=fig, axs=ax, show=False, xtick_fontsize = 13, ytick_fontsize=14, xtick_rotation=45, ytick_rotation=50)
    assert all(label.get_fontsize() == 13. for label in ax.get_xticklabels())
    assert all(label.get_fontsize() == 14. for label in ax.get_yticklabels())
    assert all(label.get_rotation() == 45 for label in ax.get_xticklabels())
    assert all(label.get_rotation() == 50 for label in ax.get_yticklabels())
    plt.close()

def test_render_plot_show():
    fig, ax = plt.subplots()
    returned_value = crested.pl.render_plot(fig=fig, axs=ax, show=True)
    assert returned_value is None
    plt.close()

def test_render_plot_saving(tmp_path):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, 5), np.arange(1, 6))
    crested.pl.render_plot(fig=fig, axs=ax, save_path=tmp_path/'test_render_plot_saving.png')
    plt.close()

# ---------- Test qc -------------
def test_qc_normalization_weights():
    regions = [f"chr{chr_i}:{start}-{start+100}" for chr_i in range(10) for start in range(100, 2000, 100)]
    adata = create_anndata_with_regions(regions, random_state=42)
    crested.pp.normalize_peaks(adata, gini_std_threshold = 0.1, top_k_percent=0.5)
    fig, ax = crested.pl.qc.normalization_weights(
        adata=adata,
        plot_kws={'color': 'red'},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

def test_qc_sort_cutoff(adata):
    fig, ax = crested.pl.qc.filter_cutoff(
        adata=adata,
        plot_kws={'alpha': 0.5},
        line_kws={'alpha': 0.5},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

def test_qc_sort_and_filter_cutoff(adata):
    # Test gini
    fig, ax = crested.pl.qc.sort_and_filter_cutoff(
        adata=adata,
        cutoffs=[300, 450],
        method='gini',
        plot_kws={'alpha': 0.5},
        line_kws={'alpha': 0.5},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

    # Test proportion
    fig, ax = crested.pl.qc.sort_and_filter_cutoff(
        adata=adata,
        cutoffs=[300, 450],
        method='proportion',
        plot_kws={'alpha': 0.5},
        line_kws={'alpha': 0.5},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

# ---------- Test region -------------

def test_region_bar_all(adata_preds):
    # Plot all models
    fig, axs = crested.pl.region.bar(
        adata_preds,
        region="chr1:194208032-194208532",
        model_names=None,
        log_transform=True,
        pred_color='pink',
        truth_color='purple',
        plot_kws={'alpha': 0.5},
        show=False
    )
    assert len(axs) == 3
    assert fig is not None and axs is not None
    plt.close()

def test_region_bar_single(adata_preds):
    # Plot one model
    fig, ax = crested.pl.region.bar(
        adata_preds,
        region="chr1:194208032-194208532",
        model_names='model_1',
        show=False
    )
    assert isinstance(ax, plt.Axes)
    assert fig is not None and ax is not None
    plt.close()

    # Plot ground truth
    fig, ax = crested.pl.region.bar(
        adata_preds,
        region="chr1:194208032-194208532",
        model_names='truth',
        show=False
    )
    assert isinstance(ax, plt.Axes)
    assert fig is not None and ax is not None
    plt.close()

def test_region_bar_prediction():
    # Plot manual prediction
    prediction = np.abs(np.random.randn(19))
    classes = [f"class_{i}" for i in range(19)]
    fig, ax = crested.pl.region.bar(
        data=prediction,
        classes=classes,
        plot_kws={'alpha': 0.5},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

def test_region_scatter(adata_preds):
    fig, ax = crested.pl.region.scatter(
        adata_preds,
        adata_preds.var_names[0],
        model_names='model_1',
        log_transform=True,
        identity_line=True,
        show=False
    )

    assert fig is not None and ax is not None
    plt.close()

# ---------- Test corr -------------

def test_corr_heatmap_self(adata_preds):
    # Default function
    fig, ax = crested.pl.corr.heatmap_self(
        adata=adata_preds,
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

    # Changed parameters
    fig, ax = crested.pl.corr.heatmap_self(
        adata=adata_preds,
        log_transform=True,
        vmin=0.,
        vmax=1.,
        reorder=True,
        cmap='coolwarm',
        cbar=True,
        cbar_kws={'shrink': 0.8},
        plot_kws={'linecolor': 'gray', 'linewidths': 0.5},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

def test_corr_heatmap_pred(adata_preds):
    # Plot single model with default-ish plot
    fig, ax = crested.pl.corr.heatmap(
        adata=adata_preds,
        model_names='model_1',
        split='test',
        cbar=False,
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

    # Plot all models with very custom plot
    fig, axs = crested.pl.corr.heatmap(
        adata=adata_preds,
        model_names=None,
        split=None,
        log_transform=True,
        vmin=0.1,
        vmax=0.8,
        reorder=True,
        cmap='viridis',
        cbar=True,
        cbar_kws={'shrink': 0.8},
        plot_kws={'linecolor': 'gray', 'linewidths': 0.5},
        show=False
    )
    assert len(axs) == 2
    assert fig is not None and axs is not None
    plt.close()


def test_corr_scatter(adata_preds):
    # Plot single class for two models without density
    fig, ax = crested.pl.corr.scatter(
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

    # Plot all classes for single model with density
    fig, ax = crested.pl.corr.scatter(
        adata=adata_preds,
        model_names='model_1',
        split="test",
        log_transform=True,
        exclude_zeros=True,
        density_indication=True,
        square=True,
        identity_line=True,
        cbar=True,
        downsample_density=40, # Since adata only has 100 points
        max_threads=1, # Not sure how many threads we have on the CI runner
        plot_kws={'alpha': 0.5},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()


def test_corr_violin(adata_preds):
    fig, ax = crested.pl.corr.violin(
        adata=adata_preds,
        split=None,
        log_transform=True,
        plot_kws={'saturation': 0.5},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

# ---------- Test dist -------------
def test_dist_histogram(adata_preds):
    # Test simple plot
    fig, axs = crested.pl.dist.histogram(
        adata=adata_preds,
        split="test",
        show=False
    )
    assert len(axs) == 12
    assert fig is not None and axs is not None
    plt.close()

    # Test custom plot
    fig, ax = crested.pl.dist.histogram(
        adata=adata_preds,
        target='model_1',
        class_names=adata_preds.obs_names[2],
        split=None,
        log_transform=False,
        plot_kws={'color': 'pink'},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()


# ---------- Test explain -------------

def test_patterns_contribution_scores():
    scores = np.random.uniform(-1, 3, (1, 1, 100, 4))
    seqs_one_hot = np.eye(4)[None, np.random.randint(4, size=100)]
    # Simple plot
    fig, ax = crested.pl.explain.contribution_scores(
        scores, seqs_one_hot, show=False
    )
    assert fig is not None and ax is not None
    # Extensive plot
    fig, ax = crested.pl.explain.contribution_scores(
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
    fig, ax = crested.pl.explain.contribution_scores(
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
    fig, ax = crested.pl.explain.contribution_scores(
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

# ----------- Test locus -----------
def test_locus_track_single():
    scores = np.random.rand(100)
    range_values=('chr1', 0, 100)
    fig, ax = crested.pl.locus.track(
        scores=scores,
        class_idxs=None,
        zoom_n_bases=90,
        coordinates=range_values,
        highlight_positions=(10, 20),
        plot_kws={'alpha': 0.5},
        show=False
    )
    assert fig is not None and ax is not None
    plt.close()

def test_locus_track_multi():
    scores = np.random.rand(100, 4)
    range_values=('chr1', 0, 100)
    fig, axs = crested.pl.locus.track(
        scores=scores,
        class_idxs=[1, 2],
        coordinates=range_values,
        class_names=[f"class_{idx}" for idx in range(4)],
        plot_kws={'alpha': 0.5},
        show=False
    )
    assert len(axs) == 2
    assert fig is not None and axs is not None
    plt.close()


def test_locus_scoring_without_bigwig():
    scores = np.random.rand(100)
    range_values = (0, 100)
    gene_start = 20
    gene_end = 40

    fig, ax = crested.pl.locus.locus_scoring(
        scores=scores,
        coordinates=range_values,
        gene_start=gene_start,
        gene_end=gene_end,
        bigwig_values=None,
        bigwig_midpoints=None,
        locus_plot_kws={'color': 'red'},
        show=False,
    )
    assert fig is not None and ax is not None
    plt.close()


def test_locus_scoring_with_bigwig():
    scores = np.random.rand(100)
    range_values = (0, 100)
    gene_start = 20
    gene_end = 40
    bigwig_values = np.random.rand(50)
    bigwig_midpoints = np.linspace(0, 100, 50)
    highlight_positions = [(range_values[1]-40, range_values[1]-30), (range_values[1]-20, range_values[1]-10)]

    fig, axs = crested.pl.locus.locus_scoring(
        scores=scores,
        coordinates=range_values,
        gene_start=gene_start,
        gene_end=gene_end,
        bigwig_values=bigwig_values,
        bigwig_midpoints=bigwig_midpoints,
        highlight_positions=highlight_positions,
        locus_plot_kws={'color': 'red'},
        bigwig_plot_kws={'color': 'red'},
        highlight_kws={'color': 'blue'},
        show=False,
    )
    assert len(axs) == 2
    assert fig is not None and axs is not None
    plt.close()

# ---------- Test design -------------
# to add

# ---------- Test modisco -------------

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
    save_path = os.path.join(save_dir, "patterns_selected_instances.png")
    crested.pl.modisco.selected_instances(
        pattern_dict=all_patterns,
        idcs=pattern_indices,
        save_path=save_path
    )
    plt.close()


def test_patterns_class_instances(all_patterns, save_dir):
    save_path = os.path.join(save_dir, "patterns_class_instances.png")
    crested.pl.modisco.class_instances(
        all_patterns, idx=2, class_representative=True, save_path=save_path
    )
    plt.close()


def test_patterns_clustermap(all_patterns, all_classes, save_dir):
    pytest.importorskip("modiscolite")
    pat_seqs = crested.tl.modisco.generate_nucleotide_sequences(all_patterns)
    pattern_matrix = crested.tl.modisco.create_pattern_matrix(
        classes=all_classes, all_patterns=all_patterns, normalize=True
    )
    save_path = os.path.join(save_dir, "patterns_clustermap.png")
    crested.pl.modisco.clustermap(
        pattern_matrix,
        classes=all_classes,
        subset=["Astro", "OPC", "Oligo"],
        pat_seqs=pat_seqs,
        grid=True,
        save_path=save_path,
        height=2,
        width=20,
    )
    plt.close()

def test_patterns_similarity_heatmap(all_patterns, save_dir):
    pytest.importorskip("modiscolite")
    pytest.importorskip("memelite")
    save_path = os.path.join(save_dir, "patterns_similarity_heatmap.png")
    sim_matrix, indices = crested.tl.modisco.calculate_similarity_matrix(all_patterns)
    crested.pl.modisco.similarity_heatmap(
        sim_matrix, indices=indices, save_path=save_path
    )
    plt.close()

if __name__ == "__main__":
    pytest.main()
