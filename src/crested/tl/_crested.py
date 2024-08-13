"""Main module to handle training and testing of the model."""

from __future__ import annotations

import os
from datetime import datetime

import keras
import numpy as np
from anndata import AnnData
from loguru import logger
from tqdm import tqdm

from crested._logging import log_and_raise
from crested.tl import TaskConfig
from crested.tl._utils import (
    _weighted_difference,
    generate_motif_insertions,
    generate_mutagenesis,
    hot_encoding_to_sequence,
    one_hot_encode_sequence,
)
from crested.tl.data import AnnDataModule
from crested.tl.data._dataset import SequenceLoader

if os.environ["KERAS_BACKEND"] == "tensorflow":
    from crested.tl._explainer_tf import Explainer
elif os.environ["KERAS_BACKEND"] == "torch":
    from crested.tl._explainer_torch import Explainer


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
        Logger to use for logging. Can be "wandb" or "tensorboard" (tensorboard not implemented yet)
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
    >>> anndatamodule = AnnDataModule(anndata, genome_file="path/to/genome.fa")
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

    >>> # Make predictions and add them to anndata as a .layers attribute
    >>> trainer.predict(anndata, model_name="predictions")

    >>> # Calculate contribution scores
    >>> scores, seqs_one_hot = trainer.calculate_contribution_scores_regions(
    ...     region_idx="chr1:1000-2000",
    ...     class_names=["class1", "class2"],
    ...     method="integrated_grad",
    ... )
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
        """Initialize callbacks"""
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
        """Initialize logger"""
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
        custom_callbacks
            List of custom callbacks to use during training.
        """
        self._check_fit_params()

        callbacks = self._initialize_callbacks(
            self.save_dir,
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
                )
            # torch.Dataloader throws "repeat" warnings when using steps_per_epoch
            elif os.environ["KERAS_BACKEND"] == "torch":
                self.model.fit(
                    train_loader.data,
                    validation_data=val_loader.data,
                    epochs=epochs,
                    callbacks=callbacks,
                    shuffle=False,
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
        dense_block_name: str = "denseblock",
        **kwargs,
    ):
        """
        Perform transfer learning on the model.

        The first phase freezes all layers before the DenseBlock (if it exists, else it freezes all layers), replaces the Dense layer, and trains with a low learning rate.
        The second phase unfreezes all layers and continues training with an even lower learning rate.
        This is especially useful in topic classification to be able to transfer learn to annotated cell type labels instead of topics.

        Ensure that you load a model first using Crested.load_model() before calling this function.

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
        mixed_precision
            Enable mixed precision training.
        kwargs
            Additional keyword arguments to pass to the fit method.

        See Also
        --------
        crested.tl.Crested.fit
        """
        logger.info(
            f"First phase of transfer learning. Freezing all layers before the DenseBlock (if it exists) and adding a new Dense Layer. Training with learning rate {learning_rate_first_phase}..."
        )
        assert (
            self.model is not None
        ), "Model is not loaded. Load a model first using Crested.load_model()."

        # Get the current optimizer configuration
        old_optimizer = self.model.optimizer
        optimizer_config = old_optimizer.get_config()
        optimizer_class = type(old_optimizer)

        base_model = self.model

        # Freeze all layers before the DenseBlock or dense_out
        for layer in base_model.layers:
            if dense_block_name in layer.name:
                found_dense_block = True
                break
            layer.trainable = False  # Freeze the layer

        if not found_dense_block:
            if not any("dense_out" in layer.name for layer in base_model.layers):
                raise ValueError(
                    f"Neither '{dense_block_name}' nor 'dense_out' found in model layers."
                )
            else:
                base_model.trainable = False

        # Change the number of output units to match the new task
        old_activation_layer = base_model.layers[-1]
        old_activation = old_activation_layer.activation

        x = base_model.layers[-3].output
        new_output_units = self.anndatamodule.adata.X.shape[0]
        new_output_layer = keras.layers.Dense(
            new_output_units, name="dense_out_transfer", trainable=True
        )(x)
        new_activation_layer = keras.layers.Activation(
            old_activation, name="activation_transfer"
        )(new_output_layer)
        new_model = keras.Model(inputs=base_model.input, outputs=new_activation_layer)

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

        evaluation_metrics = self.model.evaluate(
            test_loader.data, steps=n_test_steps, return_dict=True
        )

        # Log the evaluation results
        for metric_name, metric_value in evaluation_metrics.items():
            logger.info(f"Test {metric_name}: {metric_value:.4f}")

        if return_metrics:
            return evaluation_metrics

    def predict(
        self,
        anndata: AnnData | None = None,
        model_name: str | None = None,
    ) -> None | np.ndarray:
        """
        Make predictions using the model on the full dataset

        If anndata and model_name are provided, will add the predictions to anndata as a .layers[model_name] attribute.
        Else, will return the predictions as a numpy array.

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
        else:
            return predictions

    def predict_regions(
        self,
        region_idx: list[str] | str,
    ) -> np.ndarray:
        """
        Make predictions using the model on the specified region(s)

        Parameters
        ----------
        region_idx
            List of regions for which to make predictions in the format "chr:start-end".

        Returns
        -------
        Predictions for the specified region(s) of shape (N, C)
        """
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

        Parameters
        ----------
        model : a trained TensorFlow/Keras model
        sequence : str
            A string containing a DNA sequence (A, C, G, T).

        Returns
        -------
        np.ndarray
            Predictions for the provided sequence.
        """
        # One-hot encode the sequence
        x = one_hot_encode_sequence(sequence)

        # Make prediction
        predictions = self.model.predict(x)

        return predictions

    def calculate_contribution_scores(
        self,
        class_names: list[str],
        anndata: AnnData | None = None,
        method: str = "expected_integrated_grad",
        store_in_varm: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Calculate contribution scores based on the given method for the full dataset.

        These scores can then be plotted to visualize the importance of each base in the dataset
        using :func:`~crested.pl.patterns.contribution_scores`.

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
                explainer = Explainer(self.model, class_index=class_index)
                if method == "integrated_grad":
                    scores[:, i, :, :] = explainer.integrated_grad(
                        x, baseline_type="zeros"
                    )
                elif method == "mutagenesis":
                    scores[:, i, :, :] = explainer.mutagenesis(
                        x, class_index=class_index
                    )
                elif method == "expected_integrated_grad":
                    scores[:, i, :, :] = explainer.expected_integrated_grad(
                        x, num_baseline=25
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

        Parameters
        ----------
        region_idx
            Region(s) for which to calculate the contribution scores in the format "chr:start-end".
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
                explainer = Explainer(self.model, class_index=class_index)
                if method == "integrated_grad":
                    scores[:, i, :, :] = explainer.integrated_grad(
                        x, baseline_type="zeros"
                    )
                elif method == "mutagenesis":
                    scores[:, i, :, :] = explainer.mutagenesis(
                        x, class_index=class_index
                    )
                elif method == "expected_integrated_grad":
                    scores[:, i, :, :] = explainer.expected_integrated_grad(
                        x, num_baseline=25
                    )
                else:
                    raise

            all_scores.append(scores)

        return np.concatenate(all_scores, axis=0), np.concatenate(
            all_one_hot_sequences, axis=0
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

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing regions and class information, obtained from crested.pp.sort_and_filter_regions_on_specificity.
        output_dir : str
            Directory to save the output files.
        method : str, optional
            Method to use for calculating the contribution scores.
            Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad'.
        class_names : list[str] | None, optional
            List of class names to process. If None, all class names in adata.obs_names will be processed.
        """
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
        target_class: str,
        n_sequences: int,
        patterns: dict,
        insertions_per_pattern: dict | None = None,
        return_intermediate: bool = False,
        class_penalty_weights: np.ndarray | None = None,
        no_mutation_flanks: tuple | None = None,
        target_len: int | None = None,
        preserve_inserted_motifs: bool = True,
    ) -> tuple[list(dict), list] | list:
        """
        Create synthetic enhancers for a specified class using motif implementation.

        Parameters
        ----------
        target_class
            Class name for which the enhancers will be designed for.
        n_sequences
            Number of enhancers to design.
        patterns
            Dictionary of patterns to be implemented in the form of 'pattern_name':'pattern_sequence'
        insertions_per_pattern
            Dictionary of number of patterns to be implemented in the form of 'pattern_name':number_of_insertions
            If not used one of each pattern in patterns will be implemented.
        return_intermediate
            If True, returns a dictionary with predictions and changes made in intermediate steps for selected
            sequences
        class_penalty_weights
            Array with a value per class, determining the penalty weight for that class to be used in scoring
            function for sequence selection.
        no_mutation_flanks
            A tuple of integers which determine the regions in each flank to not do implementations.
        target_len
            Length of the area in the center of the sequence to make implementations, ignored if no_mutation_flanks
            is supplied.
        preserve_inserted_motifs
            If True, sequentially inserted motifs can't be inserted on previous motifs.

        Returns
        -------
        A list of designed sequences and if return_intermediate is True a list of dictionaries of intermediate
        mutations and predictions
        """
        self._check_contribution_scores_params([target_class])

        all_class_names = list(self.anndatamodule.adata.obs_names)

        target = all_class_names.index(target_class)

        # get input sequence length of the model
        seq_len = (
            self.anndatamodule.adata.var.iloc[0]["end"]
            - self.anndatamodule.adata.var.iloc[0]["start"]
        )

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
            insertions_per_pattern = {pattern_name: 1 for pattern_name in patterns}

        if preserve_inserted_motifs:
            inserted_motif_locations = np.array([])
        else:
            inserted_motif_locations = None

        # create random sequences
        random_sequences = self._create_random_sequences(
            n_sequences=n_sequences, seq_len=seq_len
        )

        designed_sequences = []
        intermediate_info_list = []

        for idx, sequence in enumerate(random_sequences):
            sequence_onehot = one_hot_encode_sequence(sequence)
            if return_intermediate:
                intermediate_info_list.append(
                    {
                        "inital_sequence": sequence,
                        "changes": [(-1, "N")],
                        "predictions": [
                            self.model.predict(sequence_onehot, verbose=False)
                        ],
                        "designed_sequence": "",
                    }
                )

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

                    mutagenesis_predictions = self.model.predict(mutagenesis)

                    # determine the best insertion site
                    best_mutation = _weighted_difference(
                        mutagenesis_predictions,
                        current_prediction,
                        target,
                        class_penalty_weights,
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
        target_class: str,
        n_mutations: int,
        n_sequences: int,
        return_intermediate: bool = False,
        class_penalty_weights: np.ndarray | None = None,
        no_mutation_flanks: tuple | None = None,
        target_len: int | None = None,
    ) -> tuple[list(dict), list] | list:
        """
        Create synthetic enhancers for a specified class using in silico evolution (ISE).

        Parameters
        ----------
        target_class
            Class name for which the enhancers will be designed for.
        n_mutations
            Number of mutations per sequence
        n_sequences
            Number of enhancers to design
        return_intermediate
            If True, returns a dictionary with predictions and changes made in intermediate steps for selected
            sequences
        class_penalty_weights
            Array with a value per class, determining the penalty weight for that class to be used in scoring
            function for sequence selection.
        no_mutation_flanks
            A tuple of integers which determine the regions in each flank to not do implementations.
        target_len
            Length of the area in the center of the sequence to make implementations, ignored if no_mutation_flanks
            is supplied.

        Returns
        -------
        A list of designed sequences and if return_intermediate is True a list of dictionaries of intermediate
        mutations and predictions
        """
        self._check_contribution_scores_params([target_class])

        all_class_names = list(self.anndatamodule.adata.obs_names)

        target = all_class_names.index(target_class)

        # get input sequence length of the model
        seq_len = (
            self.anndatamodule.adata.var.iloc[0]["end"]
            - self.anndatamodule.adata.var.iloc[0]["start"]
        )

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

        # create random sequences
        random_sequences = self._create_random_sequences(
            n_sequences=n_sequences, seq_len=seq_len
        )

        designed_sequences = []
        intermediate_info_list = []

        for idx, sequence in enumerate(random_sequences):
            sequence_onehot = one_hot_encode_sequence(sequence)
            if return_intermediate:
                intermediate_info_list.append(
                    {
                        "inital_sequence": sequence,
                        "changes": [(-1, "N")],
                        "predictions": [
                            self.model.predict(sequence_onehot, verbose=False)
                        ],
                        "designed_sequence": "",
                    }
                )

            # sequentially do mutations
            for _mutation_step in range(n_mutations):
                current_prediction = self.model.predict(sequence_onehot, verbose=False)

                # do every possible mutation
                mutagenesis = generate_mutagenesis(
                    sequence_onehot, include_original=False, flanks=no_mutation_flanks
                )
                mutagenesis_predictions = self.model.predict(mutagenesis)

                # determine the best mutation
                best_mutation = _weighted_difference(
                    mutagenesis_predictions,
                    current_prediction,
                    target,
                    class_penalty_weights,
                )

                sequence_onehot = mutagenesis[best_mutation : best_mutation + 1]

                if return_intermediate:
                    mutation_index = best_mutation // 3 + no_mutation_flanks[0]
                    changed_to = sequence_onehot[0, mutation_index, :]
                    intermediate_info_list[idx]["changes"].append(
                        (mutation_index, hot_encoding_to_sequence(changed_to))
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

    def _calculate_location_gc_frequencies(self) -> np.ndarray:
        regions = self.anndatamodule.adata.var
        sequence_loader = SequenceLoader(
            genome_file=self.anndatamodule.genome_file,
            chromsizes=None,
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
        # check if class names is a list
        if not isinstance(class_names, list):
            raise ValueError(
                "Class names should be a list of class names or an empty list (if calculating the average accross classes)."
            )

        all_class_names = list(self.anndatamodule.adata.obs_names)
        for class_name in class_names:
            if class_name not in all_class_names:
                raise ValueError(
                    f"Class name {class_name} not found in anndata.obs_names."
                )

    def __repr__(self):
        return f"Crested(data={self.anndatamodule is not None}, model={self.model is not None}, config={self.config is not None})"
