# Release Notes

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
-   {func}`crested.tl.Crested.score_gene_locus` now accepts an optional genome as input.
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

-   {func}`crested.tl.Crested.get_embeddings` now correcly updates .varm if anndata is passed instead of .obsm.
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
