# Release notes

## 1.7.0

This major update reworks and reorganises the `crested.pl` module, alongside other new features and fixes.

### Highlights
- The plotting functions have been reorganised into new modules ({mod}`pl.qc <crested.pl.qc>`, {mod}`pl.region <crested.pl.region>`, {mod}`pl.corr <crested.pl.corr>`, {mod}`pl.dist <crested.pl.dist>`, {mod}`pl.explain <crested.pl.explain>`, {mod}`pl.locus <crested.pl.locus>`, {mod}`pl.modisco <crested.pl.modisco>`, and {mod}`pl.design <crested.pl.design>`) and renamed accordingly. See [the dedicated page](api/renaming) for an overview. The old names still work as aliases.
- Existing plotting functions:
  - Almost all functions now accept an optional `ax` object to plot their data on, and will optionally return their `ax`/`axs` as well. This allows for composite plots and further easy customisation.
  - All plotting functions now take `plot_kws`, allowing customization of the underlying plotting function. All existing defaults (like dot size, color, alpha, ...) can also be adjusted with this.
  - Many plotting functions had their defaults (labels, size) improved and new customization arguments added.
  - Almost all functions now use `render_plot`, making them customizable by passing kwargs to the plotting function directly. `render_plot` itself is reworked to primarily work at the ax level, and allows easy setting of labels, titles, tick rotation, etc, per subplot (if passing a list) or for all subplots together (if passing a single value).
- New plotting functions:
  - {func}`crested.pl.qc.filter_cutoff` lets you visualize the impact of your gini specificity filtering with {func}`crested.pp.filter_regions_on_specificity`. 
  - {func}`crested.pl.qc.sort_and_filter_cutoff` lets you visualize the gini distributions of your top k regions when doing per-class specificity filtering with {func}`crested.pp.sort_and_filter_regions_on_specificity`.
  - {func}`crested.pl.region.scatter` lets you plot a single region's ground truth vs predictions as a scatterplot. 
- Almost all plotting functions are now tested. 
- Enhancer design functions are now grouped in `tl.design`. Again, see [the dedicated page](api/renaming) for an overview.
- All tutorials are updated with new plotting function names and new functionality. 
- Other new functions:
  - Added {func}`crested.tl.evaluate` function to not have to reinstate the Crested object just to get test metrics.
  - Added {func}`crested.utils.load_model`, which should solve most issues with loading models & serialized layers.
  - Added {func}`crested.utils.flip_region_strand` and {func}`crested.utils.parse_region` for region handling. 
- API documentation is now automatically generated recursively.

### Full changelog
#### Features
- All plotting functions:
  - All functions now use {func}`render_plot <crested.pl.render_plot>` (except a few `pl.patterns` modisco clustermaps)
  - (Almost) all functions now use {func}`create_plot <crested.pl.create_plot>`, which means they accept an axis to plot their data on, if plotting a single panel, and have unified width/height/sharex/sharey support.
  - (Almost) all functions now accept `plot_kws` to add and customize the underlying plotting function's arguments.
  - All functions take `width` and `height` to set dimensions, and multi-plot functions also take `sharex`/`sharey`.
  - {func}`render_plot <crested.pl.render_plot>` now can also set ax-level labels and titles, set x/y lims, control both axis and sup title/label fontsizes, and add a (nicely behaving) grid.
  -  Rotated labels now align with their ticks, optimized to some heuristics. Primarily important with longer cell type names. 
  -  Returning a plot with `show=False`  now returns both the fig and (list of) axes, instead of just the fig.
  -  Default axis labels now denote whether you're using `log_transform` or not.
  - Lots of plotting functions had their figure size, labels, etc defaults improved (see below), but the core plotting has been untouched.
  - Plotting now uses matplotlib's new recommended 'constrained' layout engine (which is set at plot creation) by default (if creating a new plot), making `fig.tight_layout()` unneeded. This should improve layouts with any suptitles, and organisation in `contribution_scores` if plotting multiple sequences.
- {func}`crested.pl.qc.filter_cutoff`  is a new plotting function to show the impact of different possible cutoffs when doing `pp.filter_regions_on_specificity`.
- {func}`crested.pl.qc.sort_and_filter_cutoff` is a new plotting function to show the gini scores of different classes when doing `pp.sort_and_filter_regions_on_specificity`, to establish the top k regions you want to take.
- {func}`crested.pl.region.scatter` is a new plotting function to show ground truth vs correlation for a single region. Especially useful instead of `region.bar` for many-class models where the bars are not easily interpretable. 
- {func}`crested.pl.locus.track` was expanded into a fully fledged function, and now supports large multi-class inputs, center zooming, and highlights. (#161)
-  {func}`crested.pl.explain.contribution_scores`  now supports plotting on genomic coordinates with `coordinates`.
-  The region-based barplots are now one multifunctional plotting function, {func}`crested.pl.region.bar`. This combines `pl.bar.region`, `pl.bar.region_predictions` and `pl.bar.prediction`. It supports both a single prediction matrix (like `bar.prediction`) as well as a just an anndata+region combo, and can show multiple models and/or the ground truth (like `bar.region_predictions`), while still of course being able to plot a single model/ground truth from anndata (like `bar.region`).
- Added {func}`crested.utils.flip_region_strand` and {func}`crested.utils.parse_region`, which make working with region strings/tuples easier. 
- Enhancer design functions (`tl.enhancer_design_*`, `utils.EnhancerOptimizer`, `utils.derive_intermediate_sequences`) are now grouped in `tl.design`.
- Added default config  `'peak_regression_count'` to represent the intended default with cut sites (the 'alternative config' shown in the tutorial)
- Removed ZeroPenaltyMetric from default configs.
- Added {func}`crested.tl.evaluate` function: calculates metrics on a set of choice (like test set), from model or saved preds. Means you no longer need to reinstate the Crested object just to get test metrics, only to ignore it in the rest of the workflow.
- Added {func}`crested.utils.load_model`, which has `compile=False` by default (fixing common issues with that), and tries to add custom layers and functions from `tl.utils.zoo` to custom_objects automatically if loading fails on the first attempt. This should solve most issues with loading models & serialized layers.
- Added aliases 'borzoi_human'/'borzoi_mouse'/'borzoiprime_human'/'borzoiprime_mouse' to {func}`crested.get_model()`
 
#### Minor plotting function improvements
- {func}`crested.pl.region.bar`: now uses a y-only grid by default, since an x-grid is superfluous with a categorical axis. Now takes `log_transform`  to transform the values before plotting.
- {func}`crested.pl.corr.heatmap`/{func}`crested.pl.corr.heatmap_self`:
  - Colormap is now customizable.
  - Colorbar now has a label to show its units (pearson correlation), indicating log1p-transformation if used.
  - Heatmaps are now square (`sns.heatmap(square=True)`) by default, and default fig size was slightly changed to make it fit a square heatmap + a colorbar well.
- {func}`crested.pl.dist.histogram`: Add nice default axis labels, including denoting log-transformation if used. Non-used plots in the plot grid (if plotting multiple classes) are now hidden by default.
- {func}`crested.pl.locus.locus_scoring` now takes separate plot_kws for both the locus and bigwig plots. Previous custom arguments are now folded into the plot_kws or render_plot kwargs. Highlights can now also be customized with highlight_kws.
- {func}`crested.pl.explain.contribution_scores`: 
  - Class labels are updated to be consistently at 70% of plot height (instead of 70% of the positive values) and at 2.5% of the plot width (instead of at x=5). For 'mutagenesis', it's at 30% by default since we expect those values to be mostly negative.
  - Input dimensions are now automatically attempted to be expanded if dimensions are missing.
  - Highlights can now be customized with `highlight_kws`.
  - y-limit sharing between sequences can now explicitly be customized with `sharey=True/False/'sequence'`.
  - Internals cleaned up, also makes some behavior more consistent.
  - Now takes `coordinates`, to plot the explainer on genomic coordinates rather than just `range(0, seq_len)`.
- {func}`crested.pl.design.step_predictions`: Spelling mistake in the arguments fixed. Now always creates a square grid of plots if supplying a lot of classes, following `hist.distributions`.
- {func}`crested.pl.design.step_contribution_scores` is fully reworked to wrap around `contribution_scores` and do all nice things `contribution_scores` can do. 
- `pl.modisco.*` (prev `patterns` stuff): 
  - All functions now take width/height, and the non-clustermap functions now all use `render_plot`. Clustermap functions now use `g.savefig()` as recommended by seaborn instead of `fig.savefig`.
  - {func}`crested.pl.modisco.selected_instances` now takes an axis if plotting a single index.
  - All clustermaps/heatmaps in this module should now have `cmap` as an argument.
- {func}`crested.pl.corr.scatter`:
  - Now has a `square`  argument, to make the subplot square and unify the axes and their aspect ratios (so that y=x is a perfect diagonal).
   - Now has an optional argument for an identity (y=x) line.
   - Now allows disabling of the colorbar (off by default).
   - Now has nicer default labels. 
- {func}`crested.pl.corr.violin`: Label adjusted if using log-transformed data.

#### Bugfixes
- {func}`crested.pl.design.step_contribution_scores` (ex-`patterns.enhancer_design_contribution_scores`)'s `zoom_n_bases` argument now works (#167)
- {func}`crested.pl.corr.scatter`'s colorbar now properly shows the color range (without `alpha` diluting the colors, and properly using the range of `z` fit in the function).
- {func}`crested.pl.modisco.clustermap_with_pwm_logos`: improved pwm positioning and sizing
- {func}`crested.tl.modisco.find_pattern_matches`: support new modisco versions using 'pval' rather than 'qval' columns, rename cutoff argument to 'p_val_cutoff'

#### Documentation and infrastructure
- All tutorials were updated with new functions and general improvements.
- (Almost) all plotting functions now have tests.
- API documentation is now automatically generated recursively, meaning everything is auto-generated and there's no longer a need to manually add new functions to toctrees or indeed adjust anything manually outside docstrings.
- Unused package imports were removed

#### Function deprecation warnings (will be removed in a future version)
- All old names in `crested.pl`, except `crested.pl.locus`.
- `grad_times_input_to_df`
- `grad_times_input_to_df_mutagenesis`
- `grad_times_input_to_df_mutagenesis_letters`

#### Function removals
- The methods of {class}`~crested.tl.Crested` which were superseded by {mod}`~crested.tl` functions in v1.3.0:
    - `get_embeddings` -> {func}`~crested.tl.extract_layer_embeddings`
    - `predict` -> {func}`~crested.tl.predict`
    - `predict_regions` -> {func}`~crested.tl.predict`
    - `predict_sequence` -> {func}`~crested.tl.predict`
    - `score_gene_locus` -> {func}`~crested.tl.score_gene_locus`
    - `calculate_contribution_scores` -> {func}`~crested.tl.contribution_scores`
    - `calculate_contribution_scores_regions` -> {func}`~crested.tl.contribution_scores`
    - `calculate_contribution_scores_sequence` -> {func}`~crested.tl.contribution_scores`
    - `calculate_contribution_scores_enhancer_design` -> {func}`~crested.tl.contribution_scores`
    - `tfmodisco_calculate_and_save_contribution_scores_sequences` -> {func}`~crested.tl.contribution_scores_specific`
    - `tfmodisco_calculate_and_save_contribution_scores` -> {func}`~crested.tl.contribution_scores_specific`
    - `enhancer_design_motif_implementation` -> {func}`~crested.tl.design.motif_insertion`
    - `enhancer_design_in_silico_evolution` -> {func}`~crested.tl.design.in_silico_evolution`
    - `_derive_intermediate_sequences` -> {func}`~crested.tl.design.derive_intermediate_sequences`
- Aliases for models that didn't properly reflect their nature:
    - `chrombpnet` -> {func}`~crested.tl.zoo.dilated_cnn`
    - `chrombpnet_decoupled` -> {func}`~crested.tl.zoo.dilated_cnn_decoupled`
- Superseded or obsolete utility functions:
    - `extract_bigwig_values_per_bp` -> {func}`~crested.utils.read_bigwig_region`
    - `get_value_from_dataframe` -> `df.loc[row_name, column_name]`


## 1.6.2

### Features
- Add a notice when lazy-loading modules (fixes [#165](https://github.com/aertslab/CREsted/issues/165))
- Add an `inplace` argument to all `crested.pp` functions

### Bugfixes
- Explicitly convert values to float32 in `crested.pl.contribution_scores` (fixes [#138](https://github.com/aertslab/CREsted/issues/138))
- Fix `crested.Genome.fetch()` not working with start value 0
- Fix bug in `crested.pl.bar.normalization_weights()` with modern versions of the AnnData package adding an extra dimension to data.
- Allow single-string entries for the chromosomes in `crested.pp.train_val_test_split`, not only lists.
- Point to versioned documentation of some external packages rather than latest. Mostly relevant for pandas' new update to 3.0.0. (fixes [#181](https://github.com/aertslab/CREsted/issues/181))
- Use `crested[motif]` during tests for python versions that support it.


## 1.6.1

Version fixing for scanpy (>=1.10) and python (<3.14).

## 1.6.0

From now on, crested requires a python>=3.11 installation.

### Features

- It's now possible to import crested without having a tf or torch installation. This is useful if you only need the pp, io, datasets, or plotting functionality.
  A missing backend error is only shown on `crested.tl` import.
- crested now imports lazily, meaning that the required functions only get imported when accessing them (as well as the torch/tf backend).
  This makes crested import much faster if you don't need its predicting or training functions.
- Now matches the latest scverse cookiecutter template (e.g. using "hatch" for running unittests)
- memelite dependency is moved out of default installation to an optional dependency (see README.md), as this is only used in tfmodisco with tomtom
- Store target start/end if using target_region_widths (so that there's a record in the anndata where the output values are from)
- Allow for a list/dict of files in import_bigwigs and import_beds rather than only a direct dir (to let you use files scattered across multiple directories, or a subset of bigwigs in a directory, or custom class names w/ a dict)
- Add point downsampling and parallelization to scatter.class_density()
- Change temp file creation while importing bigwigs (allows for importing bigwigs in parallel)

### Bugfixes

- now use public function keras.config.backend() rather than keras.src.backend.config.backend() everywhere
- Always use region names in crested.tl.predict to unify with model training side
- Make get_model() and get_dataset() case-insensitive
- Check for mismatches between genome, bed file, and bigwig files when reading in peak heights or tracks from bigwigs.
- Raise warning if negative values are detected when reading in tracks or peak heights
- Allow for use of a single string for a single model name in all plotting functions that take model_name (this previously gave a hard-to-parse error, since it looped over the individual letters in the string)
- Add serialisation decorators to custom attention classes

### Documentation

- Updated preprint citations to published papers
- Updated CREsted citation from Zenodo to preprint
- Added explicit data source block for models trained pretty directly on external datasets
- Clarified tissue of BICCN models
- Removed Chrombpnet citation in dilatedCNN documentation (by request of the authors)
- Add tutorial for working with custom models and explain keras 3 flexibility in CREsted.
- Fixed other small typos etc

## 1.5.0

### Features

-   Faster sequence loading (updated `crested.Genome` class and \_transform_input)
-   Better handling of very large inputs in `crested.tl.predict` by using a generator instead of copying everything to GPU first.
-   Added options for window shuffling ISM as well as option to plot ISM as letters
-   memelite update and removed tangermeme dependency
-   Add bbox_inches=tight to render_plot (small breaking)
-   New validation logic for highlight_positions parameters in `locus_scoring` to make errors more clear.

## 1.4.1

Small release with the new models for the Hydropv2 papers.

### Features

-   new models in the model repository from the Hydropv2 paper
    -   Embryo10x
    -   EmbryoHydrop
    -   MouseCortexHydrop
-   new argument option in `crested.pl.heatmap.correlations_self` and `crested.pl.heatmap.correlations_predictions` to order plot on class similarity

## 1.4.0

This is the version that is released together with the preprint of the CREsted paper.

### Features

-   new models in the model repository from the CREsted paper
    -   DeepCCL
    -   DeepGlioma
    -   DeepPBMC
    -   DeepZebraFish
    -   BorzoiBICCN
    -   DeepBICCN2
-   Some new flexibility to the Borzoi model architectures.
-   Chrombpnet architecture is renamed to DilatedCNN to avoid confusion with the Chrombpnet framework (with backwards compatibility).
-   Updates to contribution scores calculations to reduce memory consumption and clean up function structure.
-   Now possible to choose a custom model save path when calling `fit(...)`.
-   Borzoi peak regression notebook added.
-   New `crested.pl.violin` plotting module and added `crested.pl.violin.correlations`.

### Bugfixes

-   Fixed a bug that causes models to not be serialized correctly during saving, making them unable to load in compile=True mode.

## 1.3.0

This is a big release wherein we introduce our model repository and do a functional refactoring of our `tl.Crested` class.

### Features

-   new `crested.get_model` function to fetch models from crested model repository
-   new `enformer` and `borzoi` model options in the crested zoo, as well as scripts for converting weights to keras format.
-   new cut site bigwigs option for mouse cortex dataset
-   new crested model repository in the readthedocs with model descriptions.
-   new pattern clustering plot `pl.patterns.clustermap_with_pwm_logos` that shows PWM logo plots below the heatmap.
-   Extra parameters options in modisco calculations and plotting regarding allowed seqlets per cluster and top_n_regions selection.
-   option to color lines in gene locus scoring plotting
-   extra ylim option in `crested.pl.bar.prediction`
-   gene locus scoring plotting improvements

### Bugfixes

-   importing bigwigs now correctly accounts for regions that were removed due to chromsizes

### Notebooks

-   Rewrote the tutorials to use the new functional API (WIP).
-   Expanded on the enhancer design section

### Functional Refactor of crested.tl.Crested(...) class

In this large refactor we're moving everything from the Crested class that does not use both a model and the AnnDatamodule out to a new \_tools.py module where everything will be functional.
All the old functions remain in the Crested class for backward compatibility (for now) but will now raise a deprecation warning.

We're giving up a bit of clarity for ease of use by combining functions that do the same on different inputs into one single function.

#### Equivalent new functions

-   `tl.Crested.get_embeddings(...)` ---> `tl.extract_layer_embeddings(...)`
-   `tl.Crested.predict(...)` --->` tl.predict(...)`
-   `tl.Crested.predict_regions(...)` ---> `tl.predict(...)`
-   `tl.Crested.predict_sequence(...)` ---> `tl.predict(...)`
-   `tl.Crested.score_gene_locus(...)` ---> `tl.score_gene_locus(...)`
-   `tl.Crested.calculate_contribution_scores(...)` ---> `tl.contribution_scores(...)`
-   `tl.Crested.calculate_contribution_scores_regions(...)` ---> `tl.contribution_scores(...)`
-   `tl.Crested.calculate_contribution_scores_sequence(...)` ---> `tl.contribution_scores(...)`
-   `tl.Crested.calculate_contribution_scores_enhancer_design(...)` ---> `tl.contribution_scores(...)`
-   `tl.Crested.tfmodisco_calculate_and_save_contribution_scores_sequences` ---> `tl.contribution_scores_specific(...)`
-   `tl.Crested.tfmodisco_calculate_and_save_contribution_scores` ---> `tl.contribution_scores(...)`
-   `tl.Crested.enhancer_design_motif_implementation` ---> `tl.enhancer_design_motif_insertion`
-   `tl.Crested.enhancer_design_in_silico_evolution` ---> `tl.enhancer_design_in_silico_evolution`

#### New functions

Some utility functions were hidden inside the Crested class but required to be made explicit.

-   `utils.calculate_nucleotide_distribution` (advised for enhancer design)
-   `utils.derive_intermediate_sequences` (required for inspecting intermediate results from enhancer design)

#### New behaviour

-   All functions that accept a model can now also accept lists of models, in which case the results will be averaged across models.
-   All functions use a similar api, namely they expect some 'input' that can be converted to a one hot encoding (sequences, region names, anndatas with region names), but now the conversion happens behind the scenes so the user doesn't have to worry about this and we don't have a separate function per input format.

## 1.2.1

### Bug Fixes

-   Fixed an issue where {func}`crested.pp.change_regions_width` would not actually check for the chromsizes if a Genome was registered.

## 1.2.0

### Features

-   {func}`crested.import_bigwigs` now allows BED files with entries for chromsomes which are not in the BigWig.
-   Better handling of bigwigs track in the backend and new bigwig track reading function {func}`crested.utils.read_bigwig_region`
-   Overall support for stranded datasets while preserving support for non stranded datasets.
-   DVC logging now available with tf backend
-   New option to choose the starting sequences for motif implementation and ISE in enhancer design.
-   `crested.tl.Crested.score_gene_locus` now accepts an optional genome as input.
-   output_activation now parameter for all models in zoo.
-   {func}`crested.utils.reverse_complement` and {func}`crested.utils.fetch_sequences` now available.
-   Spearman correlation metric implementation
-   Pattern plotting QOL updates
-   poisson losses implemented at {class}`crested.tl.losses.PoissonLoss` and {class}`crested.tl.losses.PoissonMultinomialLoss`
-   {class}`crested.Genome` and {func}`crested.register_genome` for better handling of genome files.
-   MSECosine loss now uses a multiplier parameter instead of standard multiplication

### Tutorials

-   Introductory notebook now fully reproducible

### Bug Fixes

-   `crested.tl.Crested.get_embeddings` now correctly updates .varm if anndata is passed instead of .obsm.
-   Tangermeme moved out of optional dependencies for tf vs torch breaking mismatches.
-   Fixed calculation of contribution scores with torch backend when using incompatible numpy version.
-   Fix incorrect None return in `Crested.test()`

### Breaking Changes

-   If providing the same project_name and run_name, the Crested class will now assume that you want to continue training from existing checkpoints.
-   'genome_file' argument name everywhere updated to 'genome'

## 1.1.0

### Features

-   enhancer design plotting and calculation functions in the `crested.tl.Crested` object
-   new module `crested.tl.modisco` with enhancer code analysis functionality
-   new plots for enhancer code analysis in `crested.pl.patterns`
-   new plot for sequence prediction in `crested.pl.bar`
-   hidden utility functions now public in `crested.utils` (such as one hot encoding)
-   CosineMSELog loss now default in peak_regression configs instead of CosineMSE
-   tangermeme tomtom option in tfmodisco functions
-   get_embeddings based on layer name in `crested.tl.Crested`
-   `crested.utils.EnhancerOptimizer` utility function
-   `crested.utils.permute_model` utility, useful for using with other toch packages
-   `crested.utils.extract_bigwig_values_per_bp` utility
-   improved correlation calculation speed in heatmaps

### Tutorials

-   introductory notebook is updated with new functionality
-   new enhancer code analysis tutorial

### Bug Fixes

-   transfer Learning function now doesn't expect the dense block anymore
-   bigwigs importing checks if it can open the file instead of looking for .bw extension
-   no more ambiguous names in bed importing
-   version requirement for modiscolite
-   general pattern matching code bugfixes

### Breaking Changes

-   utility functions and logging moved to `crested.utils`
-   tfmodisco functionality moved to `crested.tl.modisco`

### Other

-   general documentation fixes
-   style checks now enforced on push
-   both tf and torch backend tests on push

## 1.0.0

-   Initial release of CREsted
