"""Main module to handle training and testing of the model."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from anndata import AnnData
from loguru import logger
from tqdm import tqdm

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
        self.model = tf.keras.models.load_model(model_path, compile=compile)

    def fit(
        self,
        epochs: int = 100,
        mixed_precision: bool = False,
        model_checkpointing: bool = True,
        model_checkpointing_best_only: bool = True,
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        learning_rate_reduce: bool = True,
        learning_rate_reduce_patience: int = 5,
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
        early_stopping
            Enable early stopping.
        early_stopping_patience
            Number of epochs with no improvement after which training will be stopped.
        learning_rate_reduce
            Enable learning rate reduction.
        learning_rate_reduce_patience
            Number of epochs with no improvement after which learning rate will be reduced.
        custom_callbacks
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
        return_metrics
            Return the evaluation metrics as a dictionary.

        Returns
        -------
        Evaluation metrics as a dictionary or None if return_metrics is False.
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
        Predictions of shape (N, C)
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

    def calculate_contribution_scores(
        self,
        anndata: AnnData | None = None,
        class_names: list[str] | None = None,
        method: str = "expected_integrated_grad",
        store_in_varm: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Calculate contribution scores based on the given method for the full dataset.

        These scores can then be plotted to visualize the importance of each base in the dataset
        using :func:`~crested.pl.patterns.contribution_scores`.

        Parameters
        ----------
        anndata
            Anndata object to store the contribution scores in as a .varm[class_name] attribute.
            If None, will only return the contribution scores without storing them.
        class_names
            List of class names to calculate the contribution scores for (should match anndata.obs_names)
            If None, the contribution scores for the 'combined' class will be calculated.
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
        self._check_contribution_scores_params(class_names)
        self._check_gpu_availability()

        if self.anndatamodule.predict_dataset is None:
            self.anndatamodule.setup("predict")
        predict_loader = self.anndatamodule.predict_dataloader

        all_scores = []
        all_one_hot_sequences = []

        all_class_names = list(self.anndatamodule.adata.obs_names)

        if class_names is not None:
            n_classes = len(class_names)
            class_indices = [
                all_class_names.index(class_name) for class_name in class_names
            ]
            varm_names = class_names
        else:
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
        class_names: list[str] | None = None,
        method: str = "expected_integrated_grad",
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
            If None, the contribution scores for the 'combined' class will be calculated.
        method
            Method to use for calculating the contribution scores.
            Options are: 'integrated_grad', 'mutagenesis', 'expected_integrated_grad'.

        Returns
        -------
        Contribution scores (N, C, L, 4) and one-hot encoded sequences (N, L, 4).

        See Also
        --------
        crested.pl.patterns.contribution_scores
        """
        self._check_contrib_params(method)
        if self.anndatamodule.predict_dataset is None:
            self.anndatamodule.setup("predict")
        self._check_contribution_scores_params(class_names)

        if isinstance(region_idx, str):
            region_idx = [region_idx]

        all_scores = []
        all_one_hot_sequences = []

        all_class_names = list(self.anndatamodule.adata.obs_names)

        if class_names is not None:
            n_classes = len(class_names)
            class_indices = [
                all_class_names.index(class_name) for class_name in class_names
            ]
        else:
            n_classes = 1  # 'combined' class
            class_indices = [None]

        logger.info(
            f"Calculating contribution scores for {n_classes} class(es) and {len(region_idx)} region(s)."
        )
        for region in tqdm(
            region_idx,
            desc="Region",
        ):
            sequence = self.anndatamodule.predict_dataset.sequence_loader.get_sequence(
                region
            )
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

    @staticmethod
    def _check_gpu_availability():
        """Check if GPUs are available."""
        devices = tf.config.list_physical_devices("GPU")
        if not devices:
            logger.warning("No GPUs available.")

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
    def _check_contribution_scores_params(self, class_names: list | None):
        """Check if the necessary parameters are set for the calculate_contribution_scores method."""
        if not self.model:
            raise ValueError(
                "Model not set. Please load a model from pretrained using Crested.load_model(...) before calling calculate_contribution_scores_(regions)."
            )
        if class_names is not None:
            all_class_names = list(self.anndatamodule.adata.obs_names)
            for class_name in class_names:
                if class_name not in all_class_names:
                    raise ValueError(
                        f"Class name {class_name} not found in anndata.obs_names."
                    )

    def __repr__(self):
        return f"Crested(data={self.anndatamodule is not None}, model={self.model is not None}, config={self.config is not None})"
