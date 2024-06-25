# Tools `tl`

```{eval-rst}
.. currentmodule:: crested.tl
```

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    Crested
    TaskConfig
    tfmodisco
    default_configs
```


```{toctree}
:maxdepth: 2
:hidden:

data
zoo
losses
metrics
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
    zoo.chrombpnet
    zoo.deeptopic_cnn
    zoo.simple_convnet
```

## Losses

```{eval-rst}
.. autosummary::
    losses.CosineMSELoss
```

## Metrics

```{eval-rst}
.. autosummary::
    metrics.ConcordanceCorrelationCoefficient
    metrics.PearsonCorrelation
    metrics.PearsonCorrelationLog
    metrics.ZeroPenaltyMetric
```