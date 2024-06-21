"""Main module to handle training and testing of the model."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from anndata import AnnData
from loguru import logger

from crested._logging import log_and_raise
from crested.tl import TaskConfig
from crested.tl._explainer import Explainer
from crested.tl._utils import one_hot_encode_sequence
from crested.tl.data import AnnDataModule


class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
        super().__init__()

        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
        import wandb

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        wandb.log({"lr": lr}, commit=False)


class Crested:
    """
    Main class to handle training, testing, predicting and calculation of contribution scores.

    Parameters
    ----------
    data : AnnDataModule
        AnndataModule object containing the data.
    model : tf.keras.Model
        Model architecture to use for training.
    config : TaskConfig
        Task configuration (optimizer, loss, and metrics) for use in tl.Crested.
    project_name : str
        Name of the project. Used for logging and creating output directories.
        If not provided, the default project name "CREsted" will be used.
    run_name : str
        Name of the run. Used for wandb logging and creating output directories.
        If not provided, the current date and time will be used.
    logger : str
        Logger to use for logging. Can be "wandb" or "tensorboard" (tensorboard not implemented yet)
        If not provided, no additional logging will be done.
    seed : int
        Seed to use for reproducibility.

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
    >>> scores, seqs_one_hot = trainer.calculate_contribution_scores(
    ...     region_idx="chr1:1000-2000",
    ...     class_indices=[0, 1, 2],
    ...     method="integrated_grad",
    ... )
    """

    def __init__(
        self,
        data: AnnDataModule,
        model: tf.keras.Model | None = None,
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
            tf.random.set_seed(self.seed)

    @staticmethod
    def _initialize_callbacks(
        save_dir: os.PathLike,
        model_checkpointing: bool,
        model_checkpointing_best_only: bool,
        early_stopping: bool,
        early_stopping_patience: int | None,
        learning_rate_reduce: bool,
        learning_rate_reduce_patience: int | None,
        custom_callbacks: list | None,
    ) -> list:
        """Initialize callbacks"""
        callbacks = []
        if early_stopping:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                patience=early_stopping_patience,
                mode="min",
                monitor="val_loss",
            )
            callbacks.append(early_stopping_callback)
        if model_checkpointing:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(save_dir, "checkpoints", "{epoch:02d}.keras"),
                monitor="val_loss",
                save_best_only=model_checkpointing_best_only,
                save_freq="epoch",
            )
            callbacks.append(model_checkpoint_callback)
        if learning_rate_reduce:
            learning_rate_reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
                patience=learning_rate_reduce_patience,
                monitor="val_loss",
                factor=0.25,
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
            log_dir = os.path.join(project_name, run_name, "logs")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            callbacks.append(tensorboard_callback)
            run = None
        else:
            run = None

        return run, callbacks

    def load_model(self, model_path: os.PathLike, compile: bool = False) -> None:
        """
        Load a (pretrained) model from a file.

        Parameters
        ----------
        model_path : os.PathLike
            Path to the model file.
        compile: bool
            Compile model after loading.
        """
        self.model = tf.keras.models.load_model(model_path, compile=compile)

    def fit(
        self,
        epochs: int = 100,
        mixed_precision: bool = False,
        model_checkpointing: bool = True,
        model_checkpointing_best_only: bool = True,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        learning_rate_reduce: bool = False,
        learning_rate_reduce_patience: int = 5,
        custom_callbacks: list | None = None,
    ) -> None:
        """
        Fit the model on the training and validation set.

        Parameters
        ----------
        epochs : int
            Number of epochs to train the model.
        mixed_precision : bool
            Enable mixed precision training.
        model_checkpointing : bool
            Save model checkpoints.
        model_checkpointing_best_only : bool
            Save only the best model checkpoint.
        early_stopping : bool
            Enable early stopping.
        early_stopping_patience : int
            Number of epochs with no improvement after which training will be stopped.
        learning_rate_reduce : bool
            Enable learning rate reduction.
        learning_rate_reduce_patience : int
            Number of epochs with no improvement after which learning rate will be reduced.
        custom_callbacks : list
            List of custom callbacks to use during training.
        """
        self._check_fit_params()
        self._check_gpu_availability()

        callbacks = self._initialize_callbacks(
            self.save_dir,
            model_checkpointing,
            model_checkpointing_best_only,
            early_stopping,
            early_stopping_patience,
            learning_rate_reduce,
            learning_rate_reduce_patience,
            custom_callbacks,
        )

        run, logger_callbacks = self._initialize_logger(
            self.logger_type, self.project_name, self.run_name
        )
        if logger_callbacks:
            callbacks.extend(logger_callbacks)

        # Configure strategy and compile model
        # strategy = tf.distribute.MirroredStrategy()

        if mixed_precision:
            logger.warning(
                "Mixed precision enabled. This can lead to faster training times but sometimes causes instable training. Disable on CPU or older GPUs."
            )
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        lr_metric = LRLogger(self.config.optimizer)
        callbacks.append(lr_metric)

        # with strategy.scope():
        self.model.compile(
            optimizer=self.config.optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics,
        )

        print(self.model.summary())
        devices = tf.config.list_physical_devices("GPU")
        logger.info(f"Number of GPUs in use: {len(devices)}")

        # setup data
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
            self.model.fit(
                train_loader.data,
                validation_data=val_loader.data,
                epochs=epochs,
                steps_per_epoch=n_train_steps_per_epoch,
                validation_steps=n_val_steps_per_epoch,
                callbacks=callbacks,
            )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        finally:
            if run:
                run.finish()

    def test(self, return_metrics: bool = False) -> dict | None:
        """
        Evaluate the model on the test set.

        Parameters
        ----------
        return_metrics : bool
            Return the evaluation metrics as a dictionary.
        """
        self._check_test_params()
        self._check_gpu_availability()

        self.anndatamodule.setup("test")
        test_loader = self.anndatamodule.test_dataloader

        n_test_steps = len(test_loader)

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
    ) -> np.ndarray:
        """
        Make predictions using the model on the full dataset

        Adds the predictions to anndata as a .layers attribute.

        Parameters
        ----------
        anndata : AnnData
            Anndata object containing the data.
        model_name : str
            Name that will be used to store the predictions in anndata.layers[model_name].
        """
        self._check_predict_params(anndata, model_name)
        self._check_gpu_availability()

        if self.anndatamodule.predict_dataset is None:
            self.anndatamodule.setup("predict")
        predict_loader = self.anndatamodule.predict_dataloader

        n_predict_steps = len(predict_loader)

        predictions = self.model.predict(predict_loader.data, steps=n_predict_steps)

        # If anndata and model_name are provided, add predictions to anndata layers
        if anndata is not None and model_name is not None:
            logger.info(f"Adding predictions to anndata.layers[{model_name}].")
            anndata.layers[model_name] = predictions.T

        return predictions

    def predict_region(
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
        np.ndarray
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

    def calculate_contribution_scores(
        self,
        region_idx: str,
        class_indices: list | None = None,
        method: str = "integrated_grad",
        return_one_hot: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Calculate contribution scores based on given method for a specified region.

        These scores can then be plotted to visualize the importance of each base in the region
        using :func:`~crested.pl.contribution_scores`.

        Parameters
        ----------
        region_idx : str
            Region for which to calculate the contribution scores in the format "chr:start-end".
        class_indices : list
            List of class indices to calculate the contribution scores for.
            If None, the contribution scores for the 'combined' class will be calculated.
        method : str
            Method to use for calculating the contribution scores.
            Options are: 'integrated_grad', 'smooth_grad', 'mutagenesis', 'saliency', 'expected_integrated_grad'.
        return_one_hot : bool
            Return the one-hot encoded sequences along with the contribution scores.
        """
        if self.anndatamodule.predict_dataset is None:
            self.anndatamodule.setup("predict")

        if isinstance(region_idx, str):
            region_idx = [region_idx]

        all_scores = []
        all_one_hot_sequences = []

        for region in region_idx:
            sequence = self.anndatamodule.predict_dataset.sequence_loader.get_sequence(
                region
            )
            x = one_hot_encode_sequence(sequence)
            all_one_hot_sequences.append(x)

            if class_indices is not None:
                n_classes = len(class_indices)
            else:
                n_classes = 1  # 'combined' class
                class_indices = [None]

            scores = np.zeros(
                (x.shape[0], n_classes, x.shape[1], x.shape[2])
            )  # (N, C, W, 4)

            for i, class_index in enumerate(class_indices):
                explainer = Explainer(self.model, class_index=class_index)
                if method == "integrated_grad":
                    scores[:, i, :, :] = explainer.integrated_grad(
                        x, baseline_type="zeros"
                    )
                elif method == "smooth_grad":
                    scores[:, i, :, :] = explainer.smoothgrad(
                        x, num_samples=50, mean=0.0, stddev=0.1
                    )
                elif method == "mutagenesis":
                    scores[:, i, :, :] = explainer.mutagenesis(
                        x, class_index=class_index
                    )
                elif method == "saliency":
                    scores[:, i, :, :] = explainer.saliency_maps(x)
                elif method == "expected_integrated_grad":
                    scores[:, i, :, :] = explainer.expected_integrated_grad(
                        x, num_baseline=25
                    )

            all_scores.append(scores)

        if return_one_hot:
            return np.concatenate(all_scores, axis=0), np.concatenate(
                all_one_hot_sequences, axis=0
            )
        else:
            return np.concatenate(all_scores, axis=0)

    @staticmethod
    def _check_gpu_availability():
        """Check if GPUs are available."""
        devices = tf.config.list_physical_devices("GPU")
        if not devices:
            logger.warning("No GPUs available.")

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

    def __repr__(self):
        return f"Crested(data={self.anndatamodule is not None}, model={self.model is not None}, config={self.config is not None})"
