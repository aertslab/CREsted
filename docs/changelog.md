# Release Notes

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
