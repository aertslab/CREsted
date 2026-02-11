# v2.0.0 renaming and deprecation

With CREsted v2.0.0, we've reorganised the plotting functions. All functions are still there (and new ones have been added), but their organisation into submodules has been improved. Also, some old functionality has been removed.

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