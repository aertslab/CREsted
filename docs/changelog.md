# Release Notes

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
