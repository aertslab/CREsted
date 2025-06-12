"""Main module to handle training and testing of the model."""

from __future__ import annotations

import os
import re
import shutil
import warnings
from datetime import datetime
from typing import Any

import keras
import numpy as np
from anndata import AnnData
from loguru import logger
from pysam import FastaFile
from tqdm import tqdm

from crested.tl import TaskConfig
from crested.tl._explainer import integrated_grad, mutagenesis
from crested.tl.data import AnnDataModule
from crested.tl.data._dataset import SequenceLoader
from crested.utils import (
    EnhancerOptimizer,
    hot_encoding_to_sequence,
    one_hot_encode_sequence,
)
from crested.utils._logging import log_and_raise
from crested.utils._seq_utils import (
    generate_motif_insertions,
    generate_mutagenesis,
)
from crested.utils._utils import _weighted_difference


class Crested:
    """
    Main class to handle training, testing, predicting and calculation of contribution scores.

    Parameters
    ----------
    data
        AnndataModule object containing the data.
    model
        Model architecture to use for training.
    config
        Task configuration (optimizer, loss, and metrics) for use in tl.Crested.
    project_name
        Name of the project. Used for logging and creating output directories.
        If not provided, the default project name "CREsted" will be used.
    run_name
        Name of the run. Used for wandb logging and creating output directories.
        If not provided, the current date and time will be used.
    logger
        Logger to use for logging. Can be "wandb", "tensorboard", or "dvc" (tensorboard not implemented yet)
        If not provided, no additional logging will be done.
    seed
        Seed to use for reproducibility.
        WARNING: this doesn't make everything fully reproducible, especially on GPU.
        Some (GPU) operations are non-deterministic and simply can't be controlled by the seed.

    Examples
    --------
    >>> from crested.tl import Crested
    >>> from crested.tl import default_configs
    >>> from crested.tl.data import AnnDataModule
    >>> from crested.tl.zoo import deeptopic_cnn

    >>> # Load data
    >>> anndatamodule = AnnDataModule(anndata, genome="path/to/genome.fa")
    >>> model_architecture = deeptopic_cnn(seq_len=1000, n_classes=10)
    >>> configs = default_configs("topic_classification")

    >>> # Initialize trainer
    >>> trainer = Crested(
    ...     data=anndatamodule,
    ...     model=model_architecture,
    ...     config=configs,
    ...     project_name="test",
    ... )

    >>> # Fit the model
    >>> trainer.fit(epochs=100)

    >>> # Evaluate the model
    >>> trainer.test()
    """

    def __init__(
        self,
        data: AnnDataModule,
        model: keras.Model | None = None,
        config: TaskConfig | None = None,
        project_name: str | None = None,
        run_name: str | None = None,
        logger: str | None = None,
        seed: int = None,
    ):
        """Initialize the Crested object."""
        self.anndatamodule = data
        self.model = model
        self.config = config
        if project_name is None:
            project_name = "CREsted"
        self.project_name = project_name
        self.run_name = (
            run_name if run_name else datetime.now().strftime("%Y-%m-%d_%H:%M")
        )
        self.logger_type = logger

        self.seed = seed
        self.save_dir = os.path.join(self.project_name, self.run_name)
        self._check_continued_training()  # check if continuing training from a previous run
        if self.seed:
            keras.utils.set_random_seed(self.seed)
        self.devices = self._check_gpu_availability()

        self.acgt_distribution = None

    @staticmethod
    def _initialize_callbacks(
        save_dir: os.PathLike,
        model_checkpointing: bool,
        model_checkpointing_best_only: bool,
        model_checkpointing_metric: str,
        model_checkpointing_mode: str,
        early_stopping: bool,
        early_stopping_patience: int | None,
        early_stopping_metric: str,
        early_stopping_mode: str,
        learning_rate_reduce: bool,
        learning_rate_reduce_patience: int | None,
        learning_rate_reduce_metric: str,
        learning_rate_reduce_mode: str,
        custom_callbacks: list | None,
    ) -> list:
        """Initialize callbacks."""
        callbacks = []
        if early_stopping:
            early_stopping_callback = keras.callbacks.EarlyStopping(
                patience=early_stopping_patience,
                mode=early_stopping_mode,
                monitor=early_stopping_metric,
            )
            callbacks.append(early_stopping_callback)
        if model_checkpointing:
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(save_dir, "checkpoints", "{epoch:02d}.keras"),
                monitor=model_checkpointing_metric,
                save_best_only=model_checkpointing_best_only,
                mode=model_checkpointing_mode,
                save_freq="epoch",
            )
            callbacks.append(model_checkpoint_callback)
        if learning_rate_reduce:
            learning_rate_reduce_callback = keras.callbacks.ReduceLROnPlateau(
                patience=learning_rate_reduce_patience,
                monitor=learning_rate_reduce_metric,
                factor=0.25,
                mode=learning_rate_reduce_mode,
                min_lr=1e-6,
            )
            callbacks.append(learning_rate_reduce_callback)
        if custom_callbacks is not None:
            callbacks.extend(custom_callbacks)
        return callbacks

    @staticmethod
    def _initialize_logger(logger_type: str | None, project_name: str, run_name: str):
        """Initialize logger."""
        callbacks = []
        if logger_type == "wandb":
            if os.environ["KERAS_BACKEND"] != "tensorflow":
                raise ValueError(
                    "Wandb logging is only available with the tensorflow backend until wandb has finished their keras 3.0 integration."
                )
            import wandb
            from wandb.integration.keras import WandbMetricsLogger

            run = wandb.init(
                project=project_name,
                name=run_name,
            )
            wandb_callback_epoch = WandbMetricsLogger(log_freq="epoch")
            wandb_callback_batch = WandbMetricsLogger(log_freq=10)
            callbacks.extend([wandb_callback_epoch, wandb_callback_batch])
        elif logger_type == "tensorboard":
            if os.environ["KERAS_BACKEND"] != "tensorflow":
                raise ValueError("Tensorboard requires a tensorflow installation")
            log_dir = os.path.join(project_name, run_name, "logs")
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir, update_freq=10
            )
            callbacks.append(tensorboard_callback)
            run = None
        elif logger_type == "dvc":
            if os.environ["KERAS_BACKEND"] != "tensorflow":
                raise ValueError("DVC Keras logging requires a tensorflow backend")
            logger.warning("Using DVC logger. Make sure to have dvclive installed.")
            from dvclive.keras import DVCLiveCallback

            log_dir = os.path.join("logs", project_name, run_name)
            dvc_callback = DVCLiveCallback()
            callbacks.append(dvc_callback)
            run = None
        else:
            run = None

        return run, callbacks

    def load_model(self, model_path: os.PathLike, compile: bool = True) -> None:
        """
        Load a (pretrained) model from a file.

        Parameters
        ----------
        model_path
            Path to the model file.
        compile
            Compile the model after loading. Set to False if you only want to load
            the model weights (e.g. when finetuning a model). If False, you should
            provide a TaskConfig to the Crested object before calling fit.
        """
        if compile and self.config is not None:
            logger.warning(
                "Loading a model with compile=True. The CREsted config object will be ignored."
            )
        self.model = keras.models.load_model(model_path, compile=compile)

    def fit(
        self,
        epochs: int = 100,
        mixed_precision: bool = False,
        model_checkpointing: bool = True,
        model_checkpointing_best_only: bool = True,
        model_checkpointing_metric: str = "val_loss",
        model_checkpointing_mode: str = "min",
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "val_loss",
        early_stopping_mode: str = "min",
        learning_rate_reduce: bool = True,
        learning_rate_reduce_patience: int = 5,
        learning_rate_reduce_metric: str = "val_loss",
        learning_rate_reduce_mode: str = "min",
        save_dir: str | None = None,
        custom_callbacks: list | None = None,
    ) -> None:
        """
        Fit the model on the training and validation set.

        Parameters
        ----------
        epochs
            Number of epochs to train the model.
        mixed_precision
            Enable mixed precision training.
        model_checkpointing
            Save model checkpoints.
        model_checkpointing_best_only
            Save only the best model checkpoint.
        model_checkpointing_metric
            Metric to monitor to choose best models.
        model_checkpointing_mode
           'max' if a high metric is better, 'min' if a low metric is better
        early_stopping
            Enable early stopping.
        early_stopping_patience
            Number of epochs with no improvement after which training will be stopped.
        early_stopping_metric
            Metric to monitor for early stopping.
        early_stopping_mode
            'max' if a high metric is better, 'min' if a low metric is better
        learning_rate_reduce
            Enable learning rate reduction.
        learning_rate_reduce_patience
            Number of epochs with no improvement after which learning rate will be reduced.
        learning_rate_reduce_metric
            Metric to monitor for reducing the learning rate.
        learning_rate_reduce_mode
            'max' if a high metric is better, 'min' if a low metric is better
        save_dir
            Directory for saving model to. Default to project name.
        custom_callbacks
            List of custom callbacks to use during training.
        """
        self._check_fit_params()

        if save_dir is None:
            save_dir = self.save_dir

        callbacks = self._initialize_callbacks(
            save_dir,
            model_checkpointing,
            model_checkpointing_best_only,
            model_checkpointing_metric,
            model_checkpointing_mode,
            early_stopping,
            early_stopping_patience,
            early_stopping_metric,
            early_stopping_mode,
            learning_rate_reduce,
            learning_rate_reduce_patience,
            learning_rate_reduce_metric,
            learning_rate_reduce_mode,
            custom_callbacks,
        )

        run, logger_callbacks = self._initialize_logger(
            self.logger_type, self.project_name, self.run_name
        )
        if logger_callbacks:
            callbacks.extend(logger_callbacks)

        if mixed_precision:
            logger.warning(
                "Mixed precision enabled. This can lead to faster training times but sometimes causes instable training. Disable on CPU or older GPUs."
            )
            keras.mixed_precision.set_global_policy("mixed_float16")

        if self.model and (
            not hasattr(self.model, "optimizer") or self.model.optimizer is None
        ):
            self.model.compile(
                optimizer=self.config.optimizer,
                loss=self.config.loss,
                metrics=self.config.metrics,
            )

        print(self.model.summary())

        # setup data
        if (
            self.anndatamodule.train_dataset is None
            or self.anndatamodule.val_dataset is None
        ):
            self.anndatamodule.setup("fit")
        train_loader = self.anndatamodule.train_dataloader
        val_loader = self.anndatamodule.val_dataloader

        n_train_steps_per_epoch = len(train_loader)
        n_val_steps_per_epoch = len(val_loader)

        if run:
            run.config.update(self.config.to_dict())
            run.config.update(
                {
                    "epochs": epochs,
                    "n_train": len(self.anndatamodule.train_dataset),
                    "n_val": len(self.anndatamodule.val_dataset),
                    "batch_size": self.anndatamodule.batch_size,
                    "n_train_steps_per_epoch": n_train_steps_per_epoch,
                    "n_val_steps_per_epoch": n_val_steps_per_epoch,
                    "seq_len": self.anndatamodule.train_dataset.seq_len,
                    "in_memory": self.anndatamodule.in_memory,
                    "random_reverse_complement": self.anndatamodule.random_reverse_complement,
                    "max_stochastic_shift": self.anndatamodule.max_stochastic_shift,
                    "shuffle": self.anndatamodule.shuffle,
                    "mixed_precision": mixed_precision,
                    "model_checkpointing": model_checkpointing,
                    "model_checkpointing_best_only": model_checkpointing_best_only,
                    "early_stopping": early_stopping,
                    "early_stopping_patience": early_stopping_patience,
                    "learning_rate_reduce": learning_rate_reduce,
                    "learning_rate_reduce_patience": learning_rate_reduce_patience,
                }
            )

        try:
            if os.environ["KERAS_BACKEND"] == "tensorflow":
                self.model.fit(
                    train_loader.data,
                    validation_data=val_loader.data,
                    epochs=epochs,
                    steps_per_epoch=n_train_steps_per_epoch,
                    validation_steps=n_val_steps_per_epoch,
                    callbacks=callbacks,
                    shuffle=False,
                    initial_epoch=self.max_epoch,
                )
            # torch.Dataloader throws "repeat" warnings when using steps_per_epoch
            elif os.environ["KERAS_BACKEND"] == "torch":
                self.model.fit(
                    train_loader.data,
                    validation_data=val_loader.data,
                    epochs=epochs,
                    callbacks=callbacks,
                    shuffle=False,
                    initial_epoch=self.max_epoch,
                )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        finally:
            if run:
                run.finish()

    def transferlearn(
        self,
        epochs_first_phase: int = 50,
        epochs_second_phase: int = 50,
        learning_rate_first_phase: float = 1e-4,
        learning_rate_second_phase: float = 1e-6,
        freeze_until_layer_name: str | None = None,
        freeze_until_layer_index: int | None = None,
        set_output_activation: str | None = None,
        **kwargs,
    ):
        """
        Perform transfer learning on the model.

        The first phase freezes layers up to a specified layer (if provided), removes the later layers, adds a dense output layer, and trains with a low learning rate.
        The second phase unfreezes all layers and continues training with an even lower learning rate.

        Ensure that you load a model first using Crested.load_model() before calling this function and have a datamodule and config loaded in your Crested object.

        One of freeze_until_layer_name or freeze_until_layer_index must be provided.

        Parameters
        ----------
        epochs_first_phase
            Number of epochs to train in the first phase.
        epochs_second_phase
            Number of epochs to train in the second phase.
        learning_rate_first_phase
            Learning rate for the first phase.
        learning_rate_second_phase
            Learning rate for the second phase.
        freeze_until_layer_name
            Name of the layer up to which to freeze layers. If None, defaults to freezing all layers except the last layer.
        freeze_until_layer_index
            Index of the layer up to which to freeze layers. If None, defaults to freezing all layers except the last layer.
        set_output_activation
            Set output activation if different from the previous model.
        kwargs
            Additional keyword arguments to pass to the fit method.

        See Also
        --------
        crested.tl.Crested.fit
        """
        logger.info(
            f"First phase of transfer learning. Freezing all layers before the specified layer and adding a new Dense Layer. Training with learning rate {learning_rate_first_phase}..."
        )
        assert (
            self.model is not None
        ), "Model is not loaded. Load a model first using Crested.load_model()."

        # Get the current optimizer configuration
        old_optimizer = self.model.optimizer
        optimizer_config = old_optimizer.get_config()
        optimizer_class = type(old_optimizer)

        base_model = self.model

        # Freeze layers up to specified layer
        if freeze_until_layer_name is not None:
            layer_names = [layer.name for layer in base_model.layers]
            if freeze_until_layer_name not in layer_names:
                raise ValueError(
                    f"Layer with name '{freeze_until_layer_name}' not found in the model."
                )
            layer_index = (
                layer_names.index(freeze_until_layer_name) + 1
            )  # Include this layer
        elif freeze_until_layer_index is not None:
            layer_index = freeze_until_layer_index + 1
        else:
            raise ValueError(
                "One of freeze_until_layer_name or freeze_until_layer_index must be provided."
            )

        # Freeze layers up to the specified layer
        for layer in base_model.layers[:layer_index]:
            layer.trainable = False

        # Remove layers after the specified layer and get the output
        truncated_model_output = base_model.layers[layer_index - 1].output

        # Get the activation function from the original model's output layer
        if isinstance(base_model.layers[-1], keras.layers.Activation):
            old_activation = base_model.layers[-1].activation
        elif hasattr(base_model.layers[-1], "activation"):
            old_activation = base_model.layers[-1].activation
        else:
            old_activation = None

        if set_output_activation is not None:
            activation = keras.activations.get(set_output_activation)
        else:
            activation = old_activation

        new_output_units = self.anndatamodule.adata.X.shape[0]

        new_output_layer = keras.layers.Dense(
            new_output_units, name="dense_out_transfer", trainable=True
        )(truncated_model_output)

        if activation is not None:
            outputs = keras.layers.Activation(activation, name="activation_transfer")(
                new_output_layer
            )
        else:
            outputs = new_output_layer

        new_model = keras.Model(inputs=base_model.input, outputs=outputs)

        new_optimizer = optimizer_class.from_config(optimizer_config)
        new_optimizer.learning_rate = learning_rate_first_phase

        # initialize new taskconfig
        self.config = TaskConfig(
            optimizer=new_optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )

        self.model = new_model
        self.fit(epochs=epochs_first_phase, **kwargs)

        logger.info(
            f"First phase of transfer learning done. Unfreezing all layers and training further with learning rate {learning_rate_second_phase}..."
        )
        for layer in self.model.layers:
            layer.trainable = True

        new_optimizer = optimizer_class.from_config(optimizer_config)
        new_optimizer.learning_rate = learning_rate_second_phase

        self.config = TaskConfig(
            optimizer=new_optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )
        self.fit(epochs=epochs_second_phase, **kwargs)

    def test(self, return_metrics: bool = False) -> dict | None:
        """
        Evaluate the model on the test set.

        Make sure to load a model first using Crested.load_model() before calling this function.
        Make sure the model is compiled before calling this function.

        Parameters
        ----------
        return_metrics
            Return the evaluation metrics as a dictionary.

        Returns
        -------
        Evaluation metrics as a dictionary or None if return_metrics is False.
        """
        self._check_test_params()

        self.anndatamodule.setup("test")
        test_loader = self.anndatamodule.test_dataloader

        n_test_steps = (
            len(test_loader) if os.environ["KERAS_BACKEND"] == "tensorflow" else None
        )
        try:
            evaluation_metrics = self.model.evaluate(
                test_loader.data, steps=n_test_steps, return_dict=True
            )
        except AttributeError as e:
            logger.error(
                "Something went wrong during evaluation. This is most likely caused by the model not being compiled.\n"
                "Please compile the model by loading the model with compile=True or manually by using crested_object.model.compile()."
            )
            logger.error(e)
            return None

        # Log the evaluation results
        for metric_name, metric_value in evaluation_metrics.items():
            logger.info(f"Test {metric_name}: {metric_value:.4f}")
        if return_metrics:
            return evaluation_metrics
        return None

    def get_embeddings(
        self,
        layer_name: str = "global_average_pooling1d",
        anndata: AnnData | None = None,
    ) -> np.ndarray:
        """
        Extract embeddings from a specified layer in the model for all regions in the dataset.

        If anndata is provided, it will add the embeddings to anndata.varm[layer_name].

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.extract_layer_embeddings()`.

        Parameters
        ----------
        anndata
            Anndata object containing the data.
        layer_name
            The name of the layer from which to extract the embeddings.

        Returns
        -------
        Embeddings of shape (N, D), where N is the number of regions in the dataset and D is the size of the embedding layer.
        """
        warnings.warn(
            "The `get_embeddings` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.extract_layer_embeddings()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if layer_name not in [layer.name for layer in self.model.layers]:
            raise ValueError(f"Layer '{layer_name}' not found in model.")
        embedding_model = keras.models.Model(
            inputs=self.model.input, outputs=self.model.get_layer(layer_name).output
        )
        if self.anndatamodule.predict_dataset is None:
            self.anndatamodule.setup("predict")
        predict_loader = self.anndatamodule.predict_dataloader
        n_predict_steps = (
            len(predict_loader) if os.environ["KERAS_BACKEND"] == "tensorflow" else None
        )
        embeddings = embedding_model.predict(predict_loader.data, steps=n_predict_steps)

        if anndata is not None:
            anndata.varm[layer_name] = embeddings
        return embeddings

    def predict(
        self,
        anndata: AnnData | None = None,
        model_name: str | None = None,
    ) -> None | np.ndarray:
        """
        Make predictions using the model on the full dataset.

        If anndata and model_name are provided, will add the predictions to anndata as a .layers[model_name] attribute.
        Else, will return the predictions as a numpy array.

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.predict()`.

        Parameters
        ----------
        anndata
            Anndata object containing the data.
        model_name
            Name that will be used to store the predictions in anndata.layers[model_name].

        Returns
        -------
        None or Predictions of shape (N, C)
        """
        warnings.warn(
            "The `predict` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.predict()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._check_predict_params(anndata, model_name)

        if self.anndatamodule.predict_dataset is None:
            self.anndatamodule.setup("predict")
        predict_loader = self.anndatamodule.predict_dataloader

        n_predict_steps = (
            len(predict_loader) if os.environ["KERAS_BACKEND"] == "tensorflow" else None
        )

        predictions = self.model.predict(predict_loader.data, steps=n_predict_steps)

        # If anndata and model_name are provided, add predictions to anndata layers
        if anndata is not None and model_name is not None:
            logger.info(f"Adding predictions to anndata.layers[{model_name}].")
            anndata.layers[model_name] = predictions.T
            return None
        else:
            return predictions

    def predict_regions(
        self,
        region_idx: list[str] | str,
    ) -> np.ndarray:
        """
        Make predictions using the model on the specified region(s).

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.predict()`.

        Parameters
        ----------
        region_idx
            List of regions for which to make predictions in the format of your original data, either "chr:start-end" or "chr:start-end:strand".

        Returns
        -------
        Predictions for the specified region(s) of shape (N, C)
        """
        warnings.warn(
            "The `predict_regions` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.predict()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.anndatamodule.predict_dataset is None:
            self.anndatamodule.setup("predict")
        if isinstance(region_idx, str):
            region_idx = [region_idx]

        all_predictions = []

        for region in region_idx:
            sequence = self.anndatamodule.predict_dataset.sequence_loader.get_sequence(
                region
            )
            x = one_hot_encode_sequence(sequence)
            predictions = self.model.predict(x)
            all_predictions.append(predictions)

        return np.concatenate(all_predictions, axis=0)

    def predict_sequence(self, sequence: str) -> np.ndarray:
        """
        Make predictions using the model on the provided DNA sequence.

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.predict()`.

        Parameters
        ----------
        sequence
            A string containing a DNA sequence (A, C, G, T).

        Returns
        -------
        Predictions for the provided sequence.
        """
        warnings.warn(
            "The `predict_sequence` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.predict()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # One-hot encode the sequence
        x = one_hot_encode_sequence(sequence)

        # Make prediction
        predictions = self.model.predict(x)

        return predictions

    def score_gene_locus(
        self,
        chr_name: str,
        gene_start: int,
        gene_end: int,
        class_name: str,
        strand: str = "+",
        upstream: int = 50000,
        downstream: int = 10000,
        window_size: int = 2114,
        central_size: int = 1000,
        step_size: int = 50,
        genome: FastaFile | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        """
        Score regions upstream and downstream of a gene locus using the model's prediction.

        The model predicts a value for the central 1000bp of each window.

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.score_gene_locus()`.

        Parameters
        ----------
        chr_name
            The chromosome name (e.g., 'chr12').
        gene_start
            The start position of the gene locus (TSS for + strand).
        gene_end
            The end position of the gene locus (TSS for - strand).
        class_name
            Output class name for prediction.
        strand
            '+' for positive strand, '-' for negative strand. Default '+'.
        upstream
            Distance upstream of the gene to score. Default 50 000.
        downstream
            Distance downstream of the gene to score. Default 10 000.
        window_size
            Size of the window to use for scoring. Default 2114.
        central_size
            Size of the central region that the model predicts for. Default 1000.
        step_size
            Distance between consecutive windows. Default 50.
        genome
            Genome of species to score locus on. If none, genome of crested class is used.

        Returns
        -------
        scores
            An array of prediction scores across the entire genomic range.
        coordinates
            An array of tuples, each containing the chromosome name and the start and end positions of the sequence for each window.
        min_loc
            Start position of the entire scored region.
        max_loc
            End position of the entire scored region.
        tss_position
            The transcription start site (TSS) position.
        """
        warnings.warn(
            "The `score_gene_locus` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.score_gene_locus()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Adjust upstream and downstream based on the strand
        if strand == "+":
            start_position = gene_start - upstream
            end_position = gene_end + downstream
            tss_position = gene_start  # TSS is at the gene_start for positive strand
        elif strand == "-":
            end_position = gene_end + upstream
            start_position = gene_start - downstream
            tss_position = gene_end  # TSS is at the gene_end for negative strand
        else:
            raise ValueError("Strand must be '+' or '-'.")

        total_length = abs(end_position - start_position)

        # Ratio to normalize the score contributions
        ratio = central_size / step_size

        # Initialize an array to store the scores, filled with zeros
        scores = np.zeros(total_length)

        # Get class index
        all_class_names = list(self.anndatamodule.adata.obs_names)
        idx = all_class_names.index(class_name)

        if genome is None:
            genome = self.anndatamodule.genome.fasta

        # Generate all windows and one-hot encode the sequences in parallel
        all_sequences = []
        all_coordinates = []

        for pos in range(start_position, end_position, step_size):
            window_start = pos
            window_end = pos + window_size

            # Ensure the window stays within the bounds of the region
            if window_end > end_position:
                break

            # Fetch the sequence
            seq = genome.fetch(chr_name, window_start, window_end).upper()

            # One-hot encode the sequence (you would need to ensure this function is available)
            seq_onehot = one_hot_encode_sequence(seq)

            all_sequences.append(seq_onehot)
            all_coordinates.append((chr_name, int(window_start), int(window_end)))

        # Stack sequences for batch processing
        all_sequences = np.squeeze(np.stack(all_sequences), axis=1)

        # Perform batched predictions
        predictions = self.model.predict(all_sequences, verbose=0)

        # Map predictions to the score array
        for _, (pos, prediction) in enumerate(
            zip(range(start_position, end_position, step_size), predictions)
        ):
            window_start = pos
            central_start = pos + (window_size - central_size) // 2
            central_end = central_start + central_size

            scores[
                central_start - start_position : central_end - start_position
            ] += prediction[idx]
            # if strand == '+':
            #    scores[central_start - start_position:central_end - start_position] += prediction[idx]
            # else:
            #    scores[total_length - (central_end - start_position):total_length - (central_start - start_position)] += prediction[idx]

        # Normalize the scores based on the number of times each position is included in the central window
        return (
            scores / ratio,
            all_coordinates,
            start_position,
            end_position,
            tss_position,
        )

    def calculate_contribution_scores(
        self,
        class_names: list[str],
        anndata: AnnData | None = None,
        method: str = "expected_integrated_grad",
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Calculate contribution scores based on the given method for the full dataset.

        These scores can then be plotted to visualize the importance of each base in the dataset
        using :func:`~crested.pl.patterns.contribution_scores`.

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.contribution_scores()`.

        Parameters
        ----------
        class_names
            List of class names to calculate the contribution scores for (should match anndata.obs_names)
            If the list is empty, the contribution scores for the 'combined' class will be calculated.
        anndata
            Anndata object to store the contribution scores in as a .varm[class_name] attribute.
            If None, will only return the contribution scores without storing them.
        method
            Method to use for calculating the contribution scores.
            Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad'.

        Returns
        -------
        Contribution scores (N, C, L, 4) and one-hot encoded sequences (N, L, 4) or None if anndata is provided.

        See Also
        --------
        crested.pl.patterns.contribution_scores
        """
        warnings.warn(
            "The `calculate_contribution_scores` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.contribution_scores()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(class_names, str):
            class_names = [class_names]
        self._check_contribution_scores_params(class_names)

        if self.anndatamodule.predict_dataset is None:
            self.anndatamodule.setup("predict")
        predict_loader = self.anndatamodule.predict_dataloader

        all_scores = []
        all_one_hot_sequences = []

        all_class_names = list(self.anndatamodule.adata.obs_names)

        if len(class_names) > 0:
            n_classes = len(class_names)
            class_indices = [
                all_class_names.index(class_name) for class_name in class_names
            ]
            varm_names = class_names
        else:
            logger.warning(
                "No class names provided. Calculating contribution scores for the 'combined' class."
            )
            n_classes = 1  # 'combined' class
            class_indices = [None]
            varm_names = ["combined"]
        logger.info(
            f"Calculating contribution scores for {n_classes} class(es) and {len(predict_loader)} batch(es) of regions."
        )

        for batch_index, (x, _) in enumerate(
            tqdm(
                predict_loader.data,
                desc="Batch",
                total=len(predict_loader),
            ),
        ):
            all_one_hot_sequences.append(x)

            scores = np.zeros(
                (x.shape[0], n_classes, x.shape[1], x.shape[2])
            )  # (N, C, W, 4)

            for i, class_index in enumerate(class_indices):
                if method == "integrated_grad":
                    scores[:, i, :, :] = integrated_grad(
                        x,
                        model=self.model,
                        num_baselines=1,
                        num_steps=25,
                        class_index=class_index,
                        baseline_type="zeros",
                        batch_size=128,
                    )
                elif method == "mutagenesis":
                    scores[:, i, :, :] = mutagenesis(
                        x, model=self.model, class_index=class_index, batch_size=128
                    )
                elif method == "expected_integrated_grad":
                    scores[:, i, :, :] = integrated_grad(
                        x,
                        model=self.model,
                        num_baselines=25,
                        num_steps=25,
                        class_index=class_index,
                        baseline_type="random",
                        batch_size=128,
                        seed=42,
                    )
            all_scores.append(scores)

            # predict_loader.data is infinite, so limit the number of iterations
            if batch_index == len(predict_loader) - 1:
                break

        concatenated_scores = np.concatenate(all_scores, axis=0)

        if anndata:
            for varm_name in varm_names:
                logger.info(f"Adding contribution scores to anndata.varm[{varm_name}].")
                if varm_name == "combined":
                    anndata.varm[varm_name] = concatenated_scores[:, 0]
                else:
                    anndata.varm[varm_name] = concatenated_scores[
                        :, class_names.index(varm_name)
                    ]
            anndata.varm["one_hot_sequences"] = np.concatenate(
                all_one_hot_sequences, axis=0
            )
            logger.info(
                "Added one-hot encoded sequences and contribution scores per class to anndata.varm."
            )
        else:
            return concatenated_scores, np.concatenate(all_one_hot_sequences, axis=0)

    def calculate_contribution_scores_regions(
        self,
        region_idx: list[str] | str,
        class_names: list[str],
        method: str = "expected_integrated_grad",
        disable_tqdm: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate contribution scores based on given method for a specified region.

        These scores can then be plotted to visualize the importance of each base in the region
        using :func:`~crested.pl.patterns.contribution_scores`.

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.contribution_scores()`.

        Parameters
        ----------
        region_idx
            Region(s) for which to calculate the contribution scores in the format "chr:start-end" or "chr:start-end:strand".
        class_names
            List of class names to calculate the contribution scores for (should match anndata.obs_names)
            If the list is empty, the contribution scores for the 'combined' class will be calculated.
        method
            Method to use for calculating the contribution scores.
            Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad'.
        disable_tqdm
            Boolean for disabling the plotting progress of calculations using tqdm.

        Returns
        -------
        Contribution scores (N, C, L, 4) and one-hot encoded sequences (N, L, 4).

        See Also
        --------
        crested.pl.patterns.contribution_scores
        """
        warnings.warn(
            "The `calculate_contribution_scores_regions` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.contribution_scores()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(region_idx, str):
            region_idx = [region_idx]

        if isinstance(class_names, str):
            class_names = [class_names]

        if self.anndatamodule.predict_dataset is None:
            self.anndatamodule.setup("predict")

        sequences = []
        for region in region_idx:
            sequences.append(
                self.anndatamodule.predict_dataset.sequence_loader.get_sequence(region)
            )
        return self.calculate_contribution_scores_sequence(
            sequences=sequences,
            class_names=class_names,
            method=method,
            disable_tqdm=disable_tqdm,
        )

    def calculate_contribution_scores_sequence(
        self,
        sequences: list[str] | str,
        class_names: list[str],
        method: str = "expected_integrated_grad",
        disable_tqdm: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate contribution scores based on given method for a specified sequence.

        These scores can then be plotted to visualize the importance of each base in the sequence
        using :func:`~crested.pl.patterns.contribution_scores`.

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.contribution_scores()`.

        Parameters
        ----------
        sequence
            Sequence(s) for which to calculate the contribution scores.
        class_names
            List of class names to calculate the contribution scores for (should match anndata.obs_names)
            If the list is empty, the contribution scores for the 'combined' class will be calculated.
        method
            Method to use for calculating the contribution scores.
            Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad'.
        disable_tqdm
            Boolean for disabling the plotting progress of calculations using tqdm.

        Returns
        -------
        Contribution scores (N, C, L, 4) and one-hot encoded sequences (N, L, 4).

        See Also
        --------
        crested.pl.patterns.contribution_scores
        """
        warnings.warn(
            "The `calculate_contribution_scores_sequence` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.contribution_scores()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(sequences, str):
            sequences = [sequences]

        if isinstance(class_names, str):
            class_names = [class_names]

        self._check_contrib_params(method)
        if self.anndatamodule.predict_dataset is None:
            self.anndatamodule.setup("predict")
        self._check_contribution_scores_params(class_names)

        all_scores = []
        all_one_hot_sequences = []

        all_class_names = list(self.anndatamodule.adata.obs_names)

        if len(class_names) > 0:
            n_classes = len(class_names)
            class_indices = [
                all_class_names.index(class_name) for class_name in class_names
            ]
        else:
            logger.warning(
                "No class names provided. Calculating contribution scores for the 'combined' class."
            )
            n_classes = 1  # 'combined' class
            class_indices = [None]

        logger.info(
            f"Calculating contribution scores for {n_classes} class(es) and {len(sequences)} region(s)."
        )
        for sequence in tqdm(sequences, desc="Region", disable=disable_tqdm):
            x = one_hot_encode_sequence(sequence)
            all_one_hot_sequences.append(x)

            scores = np.zeros(
                (x.shape[0], n_classes, x.shape[1], x.shape[2])
            )  # (N, C, W, 4)

            for i, class_index in enumerate(class_indices):
                if method == "integrated_grad":
                    scores[:, i, :, :] = integrated_grad(
                        x,
                        model=self.model,
                        num_baselines=1,
                        num_steps=25,
                        class_index=class_index,
                        baseline_type="zeros",
                        batch_size=128,
                    )
                elif method == "mutagenesis":
                    scores[:, i, :, :] = mutagenesis(
                        x,
                        model=self.model,
                        class_index=class_index,
                        batch_size=128,
                    )
                elif method == "expected_integrated_grad":
                    scores[:, i, :, :] = integrated_grad(
                        x,
                        model=self.model,
                        num_baselines=25,
                        num_steps=25,
                        class_index=class_index,
                        baseline_type="random",
                        batch_size=128,
                        seed=42,
                    )
                else:
                    raise

            all_scores.append(scores)

        return np.concatenate(all_scores, axis=0), np.concatenate(
            all_one_hot_sequences, axis=0
        )

    def calculate_contribution_scores_enhancer_design(
        self,
        enhancer_design_intermediate: list[dict],
        class_names: list[str] | None = None,
        method: str = "expected_integrated_grad",
        disable_tqdm: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | list[tuple[np.ndarray, np.ndarray]]:
        """
        Calculate contribution scores of enhancer design.

        These scores can then be plotted to visualize the importance of each base in the region
        using :func:`~crested.pl.patterns.enhancer_design_steps_contribution_scores`.

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.contribution_scores()`.

        Parameters
        ----------
        enhancer_design_intermediate
            Intermediate output from enhancer design when return_intermediate is True
        class_names
            List of class names to calculate the contribution scores for (should match anndata.obs_names)
            If None, the contribution scores for the 'combined' class will be calculated.
        method
            Method to use for calculating the contribution scores.
            Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad'.
        disable_tqdm
            Boolean for disabling the plotting progress of calculations using tqdm.

        Returns
        -------
        A tuple of arrays or a list of tuple of arrays of contribution scores (N, C, L, 4) and one-hot encoded sequences (N, L, 4).

        See Also
        --------
        crested.pl.patterns.enhancer_design_steps_contribution_scores
        crested.tl.Crested.enhancer_design_in_silico_evolution
        crested.tl.Crested.enhancer_design_motif_implementation

        Examples
        --------
        >>> scores, onehot = crested.calculate_contribution_scores_enhancer_design(
        ...     enhancer_design_intermediate,
        ...     class_names=["cell_type_A"],
        ...     method="expected_integrated_grad",
        ... )
        """
        warnings.warn(
            "The `calculate_contribution_scores_enhancer_design` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.contribution_scores()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        all_designed_list = self._derive_intermediate_sequences(
            enhancer_design_intermediate
        )

        scores_list = []
        onehot_list = []
        for designed_list in all_designed_list:
            scores, onehot = self.calculate_contribution_scores_sequence(
                sequences=designed_list,
                class_names=class_names,
                method=method,
                disable_tqdm=disable_tqdm,
            )
            scores_list.append(scores)
            onehot_list.append(onehot)

        if len(all_designed_list) == 1:
            return scores, onehot
        else:
            return scores_list, onehot_list

    def tfmodisco_calculate_and_save_contribution_scores_sequences(
        self,
        adata: AnnData,
        sequences: list[str],
        output_dir: os.PathLike = "modisco_results",
        method: str = "expected_integrated_grad",
        class_names: list[str] | None = None,
    ):
        """
        Calculate and save contribution scores for the sequence(s).

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.contribution_scores_specific()`.

        Parameters
        ----------
        adata
            The AnnData object containing class information.
        sequences:
            List of sequences (string encoded) to calculate contribution on.
        output_dir
            Directory to save the output files.
        method
            Method to use for calculating the contribution scores.
            Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad'.
        class_names
            List of class names to process. If None, all class names in adata.obs_names will be processed.
        """
        warnings.warn(
            "The `tfmodisco_calculate_and_save_contribution_scores_sequences` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.contribution_scores_specific()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Extract regions and class names from adata.var
        all_class_names = list(adata.obs_names)

        # If class_names is None, process all classes
        if class_names is None:
            class_names = all_class_names
        else:
            # Ensure that class_names contains valid classes
            valid_class_names = [
                class_name
                for class_name in class_names
                if class_name in all_class_names
            ]
            if len(valid_class_names) != len(class_names):
                raise ValueError(
                    f"Invalid class names provided. Valid class names are: {all_class_names}"
                )
            class_names = valid_class_names

        for class_name in class_names:
            # Calculate contribution scores
            contrib_scores, one_hot_seqs = self.calculate_contribution_scores_sequence(
                sequences=sequences,
                class_names=[class_name],
                method=method,
                disable_tqdm=True,
            )

            # Transform the contrib scores and one hot numpy arrays to (#regions, 4, seq_len), the expected format of modisco-lite.
            contrib_scores = contrib_scores.squeeze(axis=1).transpose(0, 2, 1)
            one_hot_seqs = one_hot_seqs.transpose(0, 2, 1)

            # Save the results to the output directory
            np.savez_compressed(
                os.path.join(output_dir, f"{class_name}_oh.npz"), one_hot_seqs
            )
            np.savez_compressed(
                os.path.join(output_dir, f"{class_name}_contrib.npz"), contrib_scores
            )

        logger.info(
            f"Contribution scores and one-hot encoded sequences saved to {output_dir}"
        )

    def tfmodisco_calculate_and_save_contribution_scores(
        self,
        adata: AnnData,
        output_dir: os.PathLike = "modisco_results",
        method: str = "expected_integrated_grad",
        class_names: list[str] | None = None,
    ):
        """
        Calculate and save contribution scores for all regions in adata.var.

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.contribution_scores_specific()`.

        Parameters
        ----------
        adata
            The AnnData object containing regions and class information, obtained from crested.pp.sort_and_filter_regions_on_specificity.
        output_dir
            Directory to save the output files.
        method
            Method to use for calculating the contribution scores.
            Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad'.
        class_names
            List of class names to process. If None, all class names in adata.obs_names will be processed.
        """
        warnings.warn(
            "The `tfmodisco_calculate_and_save_contribution_scores` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.contribution_scores()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Extract regions and class names from adata.var
        all_class_names = list(adata.obs_names)

        # If class_names is None, process all classes
        if class_names is None:
            class_names = all_class_names
        else:
            # Ensure that class_names contains valid classes
            valid_class_names = [
                class_name
                for class_name in class_names
                if class_name in all_class_names
            ]
            if len(valid_class_names) != len(class_names):
                raise ValueError(
                    f"Invalid class names provided. Valid class names are: {all_class_names}"
                )
            class_names = valid_class_names

        # If regions specificity filtering performed, then only use specific regions
        # else, use all regions in anndata.var
        if "Class name" in adata.var.columns:
            logger.info(
                "Found 'Class name' column in adata.var. Using specific regions per class to calculate contribution scores."
            )
        else:
            logger.info(
                "No 'Class name' column found in adata.var. Using all regions per class to calculate contribution scores."
            )

        for class_name in class_names:
            # Filter regions for the current class
            if "Class name" in adata.var.columns:
                class_regions = adata.var[
                    adata.var["Class name"] == class_name
                ].index.tolist()
            else:
                class_regions = adata.var.index.tolist()

            # Calculate contribution scores for the regions of the current class
            contrib_scores, one_hot_seqs = self.calculate_contribution_scores_regions(
                region_idx=class_regions,
                class_names=[class_name],
                method=method,
                disable_tqdm=True,
            )

            # Transform the contrib scores and one hot numpy arrays to (#regions, 4, seq_len), the expected format of modisco-lite.
            contrib_scores = contrib_scores.squeeze(axis=1).transpose(0, 2, 1)
            one_hot_seqs = one_hot_seqs.transpose(0, 2, 1)

            # Save the results to the output directory
            np.savez_compressed(
                os.path.join(output_dir, f"{class_name}_oh.npz"), one_hot_seqs
            )
            np.savez_compressed(
                os.path.join(output_dir, f"{class_name}_contrib.npz"), contrib_scores
            )

        logger.info(
            f"Contribution scores and one-hot encoded sequences saved to {output_dir}"
        )

    def enhancer_design_motif_implementation(
        self,
        patterns: dict,
        n_sequences: int = 1,
        target_class: str | None = None,
        target: int | np.ndarray | None = None,
        insertions_per_pattern: dict | None = None,
        return_intermediate: bool = False,
        no_mutation_flanks: tuple | None = None,
        target_len: int | None = None,
        preserve_inserted_motifs: bool = True,
        enhancer_optimizer: EnhancerOptimizer | None = None,
        starting_sequences: str | list | None = None,
        **kwargs: dict[str, Any],
    ) -> tuple[list[dict], list] | list:
        """
        Create synthetic enhancers for a specified class using motif implementation.

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.enhancer_design_motif_insertion()`.

        Parameters
        ----------
        patterns
            Dictionary of patterns to be implemented in the form of 'pattern_name':'pattern_sequence'
        n_sequences
            Number of enhancers to design.
        target_class
            Class name for which the enhancers will be designed for. If this value is set to None
            target needs to be specified.
        target
            target index, needs to be specified when target_class is None
        insertions_per_pattern
            Dictionary of number of patterns to be implemented in the form of 'pattern_name':number_of_insertions
            If not used one of each pattern in patterns will be implemented.
        return_intermediate
            If True, returns a dictionary with predictions and changes made in intermediate steps for selected
            sequences
        no_mutation_flanks
            A tuple of integers which determine the regions in each flank to not do implementations.
        target_len
            Length of the area in the center of the sequence to make implementations, ignored if no_mutation_flanks
            is supplied.
        preserve_inserted_motifs
            If True, sequentially inserted motifs can't be inserted on previous motifs.
        enhancer_optimizer
            An instance of EnhancerOptimizer, defining how sequences should be optimized.
            If None, a default EnhancerOptimizer will be initialized using `_weighted_difference`
            as optimization function.
        starting_sequences
            A DNA sequence or a list of DNA sequences that will be used instead of randomly generated
            sequences, if provided, n_sequences is ignored
        kwargs
            Keyword arguments that will be passed to the `get_best` function of the EnhancerOptimizer

        Returns
        -------
        A list of designed sequences and if return_intermediate is True a list of dictionaries of intermediate
        mutations and predictions
        """
        warnings.warn(
            "The `enhancer_design_motif_implementation` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.enhancer_design_motif_insertion()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if target_class is not None:
            self._check_contribution_scores_params([target_class])

            all_class_names = list(self.anndatamodule.adata.obs_names)

            target = all_class_names.index(target_class)

        elif target is None:
            raise ValueError(
                "`target` need to be specified when `target_class` is None"
            )

        if enhancer_optimizer is None:
            enhancer_optimizer = EnhancerOptimizer(optimize_func=_weighted_difference)

        # get input sequence length of the model
        seq_len = self.model.input_shape[1]

        # determine the flanks without changes
        if no_mutation_flanks is not None and target_len is not None:
            logger.warning(
                "Both no_mutation_flanks and target_len set, using no_mutation_flanks."
            )
        elif no_mutation_flanks is None and target_len is not None:
            if (seq_len - target_len) % 2 == 0:
                no_mutation_flanks = (
                    int((seq_len - target_len) // 2),
                    int((seq_len - target_len) // 2),
                )
            else:
                no_mutation_flanks = (
                    int((seq_len - target_len) // 2),
                    int((seq_len - target_len) // 2) + 1,
                )

        elif no_mutation_flanks is None and target_len is None:
            no_mutation_flanks = (0, 0)

        if insertions_per_pattern is None:
            insertions_per_pattern = dict.fromkeys(patterns, 1)

        if preserve_inserted_motifs:
            inserted_motif_locations = np.array([])
        else:
            inserted_motif_locations = None

        # create initial sequences
        if starting_sequences is None:
            initial_sequences = self._create_random_sequences(
                n_sequences=n_sequences, seq_len=seq_len
            )
        else:
            initial_sequences = self._parse_starting_sequences(starting_sequences)
            n_sequences = initial_sequences.shape[0]

        designed_sequences = []
        intermediate_info_list = []

        for idx, sequence in enumerate(initial_sequences):
            sequence_onehot = one_hot_encode_sequence(sequence)
            if return_intermediate:
                intermediate_info_list.append(
                    {
                        "initial_sequence": sequence,
                        "changes": [(-1, "N")],
                        "predictions": [
                            self.model.predict(sequence_onehot, verbose=False)[0]
                        ],
                        "designed_sequence": "",
                    }
                )

            if preserve_inserted_motifs:
                inserted_motif_locations = np.array([])

            # sequentially insert motifs
            for pattern_name in patterns:
                number_of_insertions = insertions_per_pattern[pattern_name]
                motif_onehot = one_hot_encode_sequence(patterns[pattern_name])
                motif_length = motif_onehot.shape[1]
                for _insertion_number in range(number_of_insertions):
                    current_prediction = self.model.predict(
                        sequence_onehot, verbose=False
                    )
                    # insert motifs at every possible location
                    mutagenesis, insertion_locations = generate_motif_insertions(
                        sequence_onehot,
                        motif_onehot,
                        flanks=no_mutation_flanks,
                        masked_locations=inserted_motif_locations,
                    )

                    mutagenesis_predictions = self.model.predict(
                        mutagenesis, verbose=False
                    )

                    # determine the best insertion site
                    best_mutation = enhancer_optimizer.get_best(
                        mutated_predictions=mutagenesis_predictions,
                        original_prediction=current_prediction,
                        target=target,
                        **kwargs,
                    )

                    sequence_onehot = mutagenesis[best_mutation : best_mutation + 1]

                    if preserve_inserted_motifs:
                        inserted_motif_locations = np.append(
                            inserted_motif_locations,
                            [
                                insertion_locations[best_mutation] + i
                                for i in range(motif_length)
                            ],
                        )

                    if return_intermediate:
                        insertion_index = insertion_locations[best_mutation]
                        changed_to = patterns[pattern_name]
                        intermediate_info_list[idx]["changes"].append(
                            (insertion_index, changed_to)
                        )
                        intermediate_info_list[idx]["predictions"].append(
                            mutagenesis_predictions[best_mutation]
                        )

            designed_sequence = hot_encoding_to_sequence(sequence_onehot)
            designed_sequences.append(designed_sequence)

            if return_intermediate:
                intermediate_info_list[idx]["designed_sequence"] = designed_sequence

        if return_intermediate:
            return intermediate_info_list, designed_sequences
        else:
            return designed_sequences

    def enhancer_design_in_silico_evolution(
        self,
        n_mutations: int,
        n_sequences: int = 1,
        target_class: str | None = None,
        target: int | np.ndarray | None = None,
        return_intermediate: bool = False,
        no_mutation_flanks: tuple | None = None,
        target_len: int | None = None,
        enhancer_optimizer: EnhancerOptimizer | None = None,
        starting_sequences: str | list | None = None,
        **kwargs: dict[str, Any],
    ) -> tuple[list[dict], list] | list:
        """
        Create synthetic enhancers for a specified class using in silico evolution (ISE).

        Warning
        -------
        This method is deprecated since version 1.3.0 and has been replaced by the standalone function :func:`~crested.tl.enhancer_design_in_silico_evolution()`.

        Parameters
        ----------
        n_mutations
            Number of iterations
        n_sequences
            Number of enhancers to design
        target_class
            Class name for which the enhancers will be designed for. If this value is set to None
            target needs to be specified.
        target
            target index, needs to be specified when target_class is None
        return_intermediate
            If True, returns a dictionary with predictions and changes made in intermediate steps for selected
            sequences
        no_mutation_flanks
            A tuple of integers which determine the regions in each flank to not do implementations.
        target_len
            Length of the area in the center of the sequence to make implementations, ignored if no_mutation_flanks
            is supplied.
        enhancer_optimizer
            An instance of EnhancerOptimizer, defining how sequences should be optimized.
            If None, a default EnhancerOptimizer will be initialized using `_weighted_difference`
            as optimization function.
        starting_sequences
            A DNA sequence or a list of DNA sequences that will be used instead of randomly generated
            sequences, if provided, n_sequences is ignored
        kwargs
            Keyword arguments that will be passed to the `get_best` function of the EnhancerOptimizer

        Returns
        -------
        A list of designed sequences and if return_intermediate is True a list of dictionaries of intermediate
        mutations and predictions as well as the designed sequences

        See Also
        --------
        crested.tl.Crested.calculate_contribution_scores_enhancer_design
        crested.utils.EnhancerOptimizer

        Examples
        --------
        >>> (
        ...     intermediate_results,
        ...     designed_sequences,
        ... ) = trained_crested_object.enhancer_design_in_silico_evolution(
        ...     target_class="cell_type_A",
        ...     n_mutations=20,
        ...     n_sequences=1,
        ...     return_intermediate=True,
        ... )
        """
        warnings.warn(
            "The `enhancer_design_in_silico_evolution` method is deprecated and will be removed from this class in a future version. "
            "Use the standalone function `tl.enhancer_design_in_silico_evolution()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.model is None:
            raise ValueError("Model should be loaded first!")

        if target_class is not None:
            self._check_contribution_scores_params([target_class])

            all_class_names = list(self.anndatamodule.adata.obs_names)

            target = all_class_names.index(target_class)

        elif target is None:
            raise ValueError(
                "`target` need to be specified when `target_class` is None"
            )

        if enhancer_optimizer is None:
            enhancer_optimizer = EnhancerOptimizer(optimize_func=_weighted_difference)

        # get input sequence length of the model
        seq_len = self.model.input_shape[1]

        # determine the flanks without changes
        if no_mutation_flanks is not None and target_len is not None:
            logger.warning(
                "Both no_mutation_flanks and target_len set, using no_mutation_flanks."
            )
        elif no_mutation_flanks is None and target_len is not None:
            if (seq_len - target_len) % 2 == 0:
                no_mutation_flanks = (
                    int((seq_len - target_len) // 2),
                    int((seq_len - target_len) // 2),
                )
            else:
                no_mutation_flanks = (
                    int((seq_len - target_len) // 2),
                    int((seq_len - target_len) // 2) + 1,
                )

        elif no_mutation_flanks is None and target_len is None:
            no_mutation_flanks = (0, 0)

        # create initial sequences
        if starting_sequences is None:
            initial_sequences = self._create_random_sequences(
                n_sequences=n_sequences, seq_len=seq_len
            )
        else:
            initial_sequences = self._parse_starting_sequences(starting_sequences)
            n_sequences = initial_sequences.shape[0]

        # initialize
        designed_sequences: list[str] = []
        intermediate_info_list: list[dict] = []

        sequence_onehot_prev_iter = np.zeros((n_sequences, seq_len, 4), dtype=np.uint8)

        # calculate total number of mutations per sequence
        _, L, A = sequence_onehot_prev_iter.shape
        start, end = 0, L
        start = no_mutation_flanks[0]
        end = L - no_mutation_flanks[1]
        TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ = (end - start) * (A - 1)

        mutagenesis = np.zeros(
            (n_sequences, TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ, seq_len, 4)
        )

        for i, sequence in enumerate(initial_sequences):
            sequence_onehot_prev_iter[i] = one_hot_encode_sequence(sequence)

        for _iter in tqdm(range(n_mutations)):
            baseline_prediction = self.model.predict(
                sequence_onehot_prev_iter, verbose=False
            )

            if _iter == 0:
                for i in range(n_sequences):
                    # initialize info
                    intermediate_info_list.append(
                        {
                            "initial_sequence": hot_encoding_to_sequence(
                                sequence_onehot_prev_iter[i]
                            ),
                            "changes": [(-1, "N")],
                            "predictions": [baseline_prediction[i]],
                            "designed_sequence": "",
                        }
                    )

            # do all possible mutations
            for i in range(n_sequences):
                mutagenesis[i] = generate_mutagenesis(
                    sequence_onehot_prev_iter[i : i + 1],
                    include_original=False,
                    flanks=no_mutation_flanks,
                )

            mutagenesis_predictions = self.model.predict(
                mutagenesis.reshape(
                    (n_sequences * TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ, seq_len, 4)
                ),
                verbose=False,
            )

            mutagenesis_predictions = mutagenesis_predictions.reshape(
                (
                    n_sequences,
                    TOTAL_NUMBER_OF_MUTATIONS_PER_SEQ,
                    mutagenesis_predictions.shape[1],
                )
            )

            for i in range(n_sequences):
                best_mutation = enhancer_optimizer.get_best(
                    mutated_predictions=mutagenesis_predictions[i],
                    original_prediction=baseline_prediction[i],
                    target=target,
                    **kwargs,
                )
                sequence_onehot_prev_iter[i] = mutagenesis[
                    i, best_mutation : best_mutation + 1, :
                ]
                if return_intermediate:
                    mutation_index = best_mutation // 3 + no_mutation_flanks[0]
                    changed_to = hot_encoding_to_sequence(
                        sequence_onehot_prev_iter[i, mutation_index, :]
                    )
                    intermediate_info_list[i]["changes"].append(
                        (mutation_index, changed_to)
                    )
                    intermediate_info_list[i]["predictions"].append(
                        mutagenesis_predictions[i][best_mutation]
                    )

        # get final sequence
        for i in range(n_sequences):
            best_mutation = enhancer_optimizer.get_best(
                mutated_predictions=mutagenesis_predictions[i],
                original_prediction=baseline_prediction[i],
                target=target,
                **kwargs,
            )

            designed_sequence = hot_encoding_to_sequence(
                mutagenesis[i, best_mutation : best_mutation + 1, :]
            )

            designed_sequences.append(designed_sequence)

            if return_intermediate:
                intermediate_info_list[i]["designed_sequence"] = designed_sequence

        if return_intermediate:
            return intermediate_info_list, designed_sequences
        else:
            return designed_sequences

    def _create_random_sequences(self, n_sequences: int, seq_len: int) -> np.ndarray:
        if self.acgt_distribution is None:
            self._calculate_location_gc_frequencies()

        random_sequences = np.empty((n_sequences), dtype=object)

        for idx_seq in range(n_sequences):
            current_sequence = []
            for idx_loc in range(seq_len):
                current_sequence.append(
                    np.random.choice(
                        ["A", "C", "G", "T"], p=list(self.acgt_distribution[idx_loc])
                    )
                )
            random_sequences[idx_seq] = "".join(current_sequence)

        return random_sequences

    def _parse_starting_sequences(self, starting_sequences) -> np.ndarray:
        if isinstance(starting_sequences, str):
            starting_sequences = [starting_sequences]

        n_sequences = len(starting_sequences)
        starting_sequences_array = np.empty((n_sequences), dtype=object)
        for idx, sequence in enumerate(starting_sequences):
            starting_sequences_array[idx] = sequence

        return starting_sequences_array

    def _calculate_location_gc_frequencies(self) -> np.ndarray:
        regions = self.anndatamodule.adata.var
        sequence_loader = SequenceLoader(
            genome=self.anndatamodule.genome,
            in_memory=True,
            always_reverse_complement=False,
            max_stochastic_shift=0,
            regions=list(regions.index),
        )
        all_sequences = list(sequence_loader.sequences.values())
        sequence_length = len(all_sequences[0])
        all_onehot_squeeze = np.array(
            [one_hot_encode_sequence(seq) for seq in all_sequences]
        ).squeeze(axis=1)
        acgt_distribution = np.sum(all_onehot_squeeze, axis=0).astype(int) / np.reshape(
            np.sum(np.sum(all_onehot_squeeze, axis=0), axis=1), (sequence_length, 1)
        ).astype(int)

        self.acgt_distribution = acgt_distribution
        return acgt_distribution

    def _derive_intermediate_sequences(self, enhancer_design_intermediate):
        all_designed_list = []
        for intermediate_dict in enhancer_design_intermediate:
            current_sequence = intermediate_dict["initial_sequence"]
            sequence_list = [current_sequence]
            for loc, change in intermediate_dict["changes"]:
                if loc == -1:
                    continue
                else:
                    current_sequence = (
                        current_sequence[:loc]
                        + change
                        + current_sequence[loc + len(change) :]
                    )
                    sequence_list.append(current_sequence)
            all_designed_list.append(sequence_list)
        return all_designed_list

    @staticmethod
    def _check_gpu_availability():
        """Check if GPUs are available."""
        if os.environ["KERAS_BACKEND"] == "torch":
            # torch backend not yet available in keras.distribution
            import torch

            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("No GPUs available, falling back to CPU.")
                return torch.device("cpu")

        elif os.environ["KERAS_BACKEND"] == "tensorflow":
            devices = keras.distribution.list_devices("gpu")
            if not devices:
                logger.warning("No GPUs available, falling back to CPU.")
                devices = keras.distribution.list_devices("cpu")
            return devices

    @log_and_raise(ValueError)
    def _check_contrib_params(self, method):
        if method not in [
            "integrated_grad",
            "smooth_grad",
            "mutagenesis",
            "saliency",
            "expected_integrated_grad",
        ]:
            raise ValueError(
                "Contribution score method not implemented. Choose out of the following options: integrated_grad, smooth_grad, mutagenesis, saliency, expected_integrated_grad."
            )

    @log_and_raise(ValueError)
    def _check_fit_params(self):
        """Check if the necessary parameters are set for the fit method."""
        if not self.model:
            raise ValueError(
                "Model not set. Please load a model from pretrained using Crested.load_model(...) or provide a model architecture with Crested(model=...) before calling fit."
            )
        if not self.config:
            logger.error(
                "Task configuration not set. Please provide a TaskConfig to Crested(config=...) before calling fit."
            )
            raise ValueError(
                "Task configuration not set. Please provide a TaskConfig to Crested(config=...) before calling fit."
            )
        if not self.project_name:
            raise ValueError(
                "Project name not set. Please provide a project name to Crested(project_name=...) before calling fit."
            )

    @log_and_raise(ValueError)
    def _check_test_params(self):
        """Check if the necessary parameters are set for the test method."""
        if not self.model:
            raise ValueError(
                "Model not set. Please load a model from pretrained using Crested.load_model(...) before calling test."
            )

    @log_and_raise(ValueError)
    def _check_predict_params(self, anndata: AnnData | None, model_name: str | None):
        """Check if the necessary parameters are set for the predict method."""
        if not self.model:
            raise ValueError(
                "Model not set. Please load a model from pretrained using Crested.load_model(...) before calling predict."
            )
        if (anndata is not None and model_name is None) or (
            anndata is None and model_name is not None
        ):
            raise ValueError(
                "Both anndata and model_name must be provided if one of them is provided."
            )

    @log_and_raise(ValueError)
    def _check_contribution_scores_params(self, class_names: list):
        """Check if the necessary parameters are set for the calculate_contribution_scores method."""
        if not self.model:
            raise ValueError(
                "Model not set. Please load a model from pretrained using Crested.load_model(...) before calling calculate_contribution_scores_(regions)."
            )

        all_class_names = list(self.anndatamodule.adata.obs_names)
        for class_name in class_names:
            if class_name not in all_class_names:
                raise ValueError(
                    f"Class name {class_name} not found in anndata.obs_names."
                )

    def _check_continued_training(self):
        """Check if the model is already trained and load existing model if so."""
        self.max_epoch = 0
        checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            # continue training or start from scratch
            pattern = re.compile(r".*\.keras")
            latest_checkpoint = None
            for file in os.listdir(checkpoint_dir):
                match = pattern.match(file)
                if match:
                    epoch = int(file.split(".")[0])
                    if epoch > self.max_epoch:
                        self.max_epoch = epoch
                        latest_checkpoint = file
            if latest_checkpoint:
                logger.warning(
                    f"Output directory {checkpoint_dir} already exists. Will continue training from epoch {self.max_epoch}."
                )
                latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                self.load_model(latest_checkpoint_path, compile=True)
            else:
                logger.warning(
                    f"Output directory {checkpoint_dir}, already exists but no trained models found. Overwriting..."
                )
                shutil.rmtree(checkpoint_dir)

    def __repr__(self):
        """Return the string representation of the object."""
        return f"Crested(data={self.anndatamodule is not None}, model={self.model is not None}, config={self.config is not None})"
