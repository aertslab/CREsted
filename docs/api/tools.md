# Tools `tl`

Training and testing of models.  
Explanation of models.  
Sequence design.   

```{eval-rst}
.. module:: crested.tl
```

```{eval-rst}
.. currentmodule:: crested
```

## Basic

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    tl.Crested
    tl.TaskConfig
    tl.tfmodisco
    tl.default_configs
```


## Data

Utility functions to prepare data for training and evaluation.  
Generally, `tl.data.AnnDataModule` is the only one that should be called directly by the user. 

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    tl.data.AnnDataModule
    tl.data.AnnDataLoader
    tl.data.AnnDataset
```

## Losses

Custom `tf.Keras.losses.Loss` functions for specific use cases.
Supply these (or your own) to a `tl.TaskConfig` to be able to use them for training.

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    tl.losses.CosineMSELoss
```

## Metrics

Custom `tf.keras.metrics.Metric` metrics for specific use cases.
Supply these (or your own) to a `tl.TaskConfig` to be able to use them for training.

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    tl.metrics.ConcordanceCorrelationCoefficient
    tl.metrics.PearsonCorrelation
    tl.metrics.PearsonCorrelationLog
    tl.metrics.ZeroPenaltyMetric
```

## Model Zoo

Custom `tf.keras.Model` definitions that have shown to work well in specific use cases.
Supply these (or your own) to `tl.Crested(...)` to use them in training.

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    tl.zoo.basenji
    tl.zoo.chrombpnet
    tl.zoo.deeptopic_cnn
    tl.zoo.simple_convnet
```