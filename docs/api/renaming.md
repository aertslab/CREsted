# v2.0.0 renaming and deprecation

With CREsted v2.0.0, we've reorganised the plotting functions, to align by _what_ they plot rather than _how_ they plot. All functions are still there, often with expanded functionality, and new ones have been added.

| Old name | New name |
| ------------- | ------------- |
| `bar.region` | {func}`region.bar <crested.pl.region.bar>` |
| `bar.region_predictions` | {func}`region.bar <crested.pl.region.bar>` |
| `bar.normalization_weights` | {func}`qc.normalization_weights <crested.pl.qc.normalization_weights>` |
| `hist.distribution` | {func}`dist.histogram <crested.pl.dist.histogram>`  |
| `heatmap.correlations_self` | {func}`corr.heatmap_self <crested.pl.corr.heatmap_self>` |
| `heatmap.correlations_predictions` | {func}`corr.heatmap <crested.pl.corr.heatmap>` |
| `locus.locus_scoring` | unchanged |
| `locus.track` | unchanged |
| `scatter.class_density` | {func}`corr.scatter <crested.pl.corr.scatter>` |
| `violin.correlations` | {func}`corr.violin  <crested.pl.corr.violin>` |
| `patterns.contribution_scores` | {func}`explain.contribution_scores <crested.pl.explain.contribution_scores>` |
| `patterns` modisco functions | {mod}`modisco.* <crested.pl.modisco>` |
| `patterns.enhancer_design_steps_contribution_scores` | {func}`design.step_contribution_scores <crested.pl.design.step_contribution_scores>` |
| `patterns.enhancer_design_steps_predictions` | {func}`design.step_predictions <crested.pl.design.step_predictions>` |


 We've also cleaned up functions and methods that have long been marked as deprecated:
 - The methods of {class}`~crested.tl.Crested` which were superseded by {mod}`~crested.tl` functions:
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
    - `enhancer_design_motif_implementation` -> {func}`~crested.tl.enhancer_design_motif_insertion`
    - `enhancer_design_in_silico_evolution` -> {func}`~crested.tl.enhancer_design_in_silico_evolution`
- Aliases for models that didn't properly reflect their nature:
    - `chrombpnet` -> {func}`~crested.tl.zoo.dilated_cnn`
    - `chrombpnet_decoupled` -> {func}`~crested.tl.zoo.dilated_cnn_decoupled`
- Superseded or obsolete utility functions:
    - `extract_bigwig_values_per_bp` -> {func}`~crested.utils.read_bigwig_region`
    - `get_value_from_dataframe` -> `df.loc[row_name, column_name]`
    