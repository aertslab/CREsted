"""Main module to handle training and testing of the model."""

from __future__ import annotations

import os
from datetime import datetime

import tensorflow as tf
from loguru import logger

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

    def _check_fit_params(self):
        """Check if the necessary parameters are set for the fit method."""
        if not self.model:
            raise ValueError(
                "Model not set. Please load a model from pretrained using Crested.load_model(...) or provide a model architecture with Crested(model=...) before calling fit."
            )
        if not self.config:
            raise ValueError(
                "Task configuration not set. Please provide a TaskConfig to Crested(config=...) before calling fit."
            )
        if not self.project_name:
            raise ValueError(
                "Project name not set. Please provide a project name to Crested(project_name=...) before calling fit."
            )

    def load_model(self, model_path: os.PathLike):
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
    ):
        """Fit the model."""
        self._check_fit_params()

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
        logger.info(f"Number of GPUs available: {len(devices)}")

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

    def test(self, return_metrics: bool = False):
        """Evaluate the model."""
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
