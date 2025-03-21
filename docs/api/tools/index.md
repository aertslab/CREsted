# Tools `tl`

```{eval-rst}
.. currentmodule:: crested.tl
```

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    Crested
    TaskConfig
    default_configs
    predict
    score_gene_locus
    contribution_scores
    contribution_scores_specific
    extract_layer_embeddings
    enhancer_design_in_silico_evolution
    enhancer_design_motif_insertion
```

```{toctree}
:maxdepth: 2
:hidden:

data
zoo
losses
metrics
modisco
```

## Data

```{eval-rst}
.. autosummary::
    data.AnnDataModule
    data.AnnDataLoader
    data.AnnDataset
```

## Model Zoo

```{eval-rst}
.. autosummary::
    zoo.basenji
    zoo.borzoi
    zoo.deeptopic_cnn
    zoo.deeptopic_lstm
    zoo.dilated_cnn
    zoo.dilated_cnn_decoupled
    zoo.enformer
    zoo.simple_convnet
```

## Losses

```{eval-rst}
.. autosummary::
    losses.CosineMSELoss
    losses.CosineMSELogLoss
```

## Metrics

```{eval-rst}
.. autosummary::
    metrics.ConcordanceCorrelationCoefficient
    metrics.PearsonCorrelation
    metrics.PearsonCorrelationLog
    metrics.ZeroPenaltyMetric
```

## Modisco

```{eval-rst}
.. autosummary::
    modisco.tfmodisco
    modisco.match_h5_files_to_classes
    modisco.process_patterns
    modisco.create_pattern_matrix
    modisco.generate_nucleotide_sequences
    modisco.pattern_similarity
    modisco.find_pattern
    modisco.find_pattern_matches
    modisco.calculate_similarity_matrix
    modisco.calculate_mean_expression_per_cell_type
    modisco.generate_html_paths
    modisco.read_motif_to_tf_file
    modisco.create_pattern_tf_dict
    modisco.create_tf_ct_matrix
```
