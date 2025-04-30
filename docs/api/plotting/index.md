# Plotting: `pl`

Functions to visualize your model's predictions or contribution scores.

```{eval-rst}
.. currentmodule:: crested.pl
```

```{toctree}
:maxdepth: 2
:hidden:

patterns
bar
hist
heatmap
locus
scatter
violin
```

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    render_plot
```

## Patterns: Contribution scores and Modisco results

```{eval-rst}
.. autosummary::
    patterns.contribution_scores
    patterns.modisco_results
    patterns.enhancer_design_steps_contribution_scores
    patterns.enhancer_design_steps_predictions
    patterns.selected_instances
    patterns.class_instances
    patterns.clustermap
    patterns.clustermap_with_pwm_logos
    patterns.clustermap_tf_motif
    patterns.tf_expression_per_cell_type
    patterns.similarity_heatmap
```

## Bar plots

```{eval-rst}
.. autosummary::
    bar.region
    bar.region_predictions
    bar.normalization_weights
```

## Distribution plots

```{eval-rst}
.. autosummary::
    hist.distribution
```

## Correlation heatmaps

```{eval-rst}
.. autosummary::
    heatmap.correlations_self
    heatmap.correlations_predictions
```

## Locus plots

```{eval-rst}
.. autosummary::
    locus.locus_scoring
    locus.track
```

## Scatter plots

```{eval-rst}
.. autosummary::
    scatter.class_density
```

## Violin plots

```{eval-rst}
.. autosummary::
    violin.correlations
```
