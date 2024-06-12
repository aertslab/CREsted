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
from crested.tl.data import AnnDataModule


class Crested:
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
        self.logger = logger
        self.seed = seed
        self.save_dir = os.path.join(self.project_name, self.run_name)

        if self.seed:
            tf.random.set_seed(self.seed)

    @staticmethod
    def _initialize_callbacks(
        save_dir: os.PathLike,
        model_checkpointing: bool,
        model_checkpointing_best_only: bool | None,
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

            logger_type = wandb.init(
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
            logger_type = None
        else:
            logger_type = None

        return logger_type, callbacks

    def load_model(self, model_path: os.PathLike) -> None:
        """Load a model from a file."""
        self.model = tf.keras.models.load_model(model_path, compile=True)

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
        """Fit the model."""
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

        logger_type, logger_callbacks = self._initialize_logger(
            self.logger, self.project_name, self.run_name
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

        self.model.fit(
            train_loader.data,
            validation_data=val_loader.data,
            epochs=epochs,
            steps_per_epoch=n_train_steps_per_epoch,
            validation_steps=n_val_steps_per_epoch,
            callbacks=callbacks,
        )

        # if self.logger_type == "wandb":
        #     self.logger.finish()

    def test(self, return_metrics: bool = False) -> dict | None:
        """Evaluate the model."""
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
        """Make predictions using the model."""
        self._check_predict_params(anndata, model_name)
        self._check_gpu_availability()

        self.anndatamodule.setup("predict")
        predict_loader = self.anndatamodule.predict_dataloader

        n_predict_steps = len(predict_loader)

        predictions = self.model.predict(predict_loader.data, steps=n_predict_steps)

        # If anndata and model_name are provided, add predictions to anndata layers
        if anndata is not None and model_name is not None:
            logger.info(f"Adding predictions to anndata.layers[{model_name}].")
            anndata.layers[model_name] = predictions.T

        return predictions

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
