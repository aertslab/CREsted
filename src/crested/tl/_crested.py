"""Main module to handle training and testing of the model."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime

import keras
from loguru import logger

from crested.tl import TaskConfig
from crested.tl.data import AnnDataModule, BaseDataWrapper
from crested.utils._logging import log_and_raise


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
        self.data = data
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
        save_dir: str | os.PathLike,
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
            if keras.config.backend() != "tensorflow":
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
            if keras.config.backend() != "tensorflow":
                raise ValueError("Tensorboard requires a tensorflow installation")
            log_dir = os.path.join(project_name, run_name, "logs")
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir, update_freq=10
            )
            callbacks.append(tensorboard_callback)
            run = None
        elif logger_type == "dvc":
            if keras.config.backend() != "tensorflow":
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

    def load_model(self, model_path: str | os.PathLike, compile: bool = True) -> None:
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
        if isinstance(self.data, BaseDataWrapper):
            train_loader = self.data.create_dataloader(split='train', augment=True, shuffle=True)
            val_loader = self.data.create_dataloader(split='val')
        else:
            if self.data.train_dataset is None or self.data.val_dataset is None:
                self.data.setup("fit")
            train_loader = self.data.train_dataloader.data
            val_loader = self.data.val_dataloader.data
        n_train_steps_per_epoch = self.data.get_config()['n_train_steps_per_epoch']
        n_val_steps_per_epoch = self.data.get_config()['n_val_steps_per_epoch']

        if run:
            # Update with TaskConfig object (optimizer, loss, metrics)
            run.config.update(self.config.to_dict())
            # Update with DataWrapper/DataModule config
            run.config.update(self.data.get_config())
            # Update with Crested object configs
            run.config.update(
                {
                    "epochs": epochs,
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
            if keras.config.backend() == "tensorflow":
                self.model.fit(
                    train_loader,
                    validation_data=val_loader,
                    epochs=epochs,
                    steps_per_epoch=n_train_steps_per_epoch,
                    validation_steps=n_val_steps_per_epoch,
                    callbacks=callbacks,
                    shuffle=False,
                    initial_epoch=self.max_epoch,
                )
            # torch.Dataloader throws "repeat" warnings when using steps_per_epoch
            elif keras.config.backend() == "torch":
                self.model.fit(
                    train_loader,
                    validation_data=val_loader,
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
        assert self.model is not None, (
            "Model is not loaded. Load a model first using Crested.load_model()."
        )

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

        new_output_units = self.data.adata.X.shape[0]

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

        if isinstance(self.data, BaseDataWrapper):
            test_loader = self.data.create_dataloader(split='test')
        else:
            if self.data.test_dataset is None:
                self.data.setup("test")
            test_loader = self.data.test_dataloader.data

        n_test_steps = (
            len(test_loader) if keras.config.backend() == "tensorflow" else None
        )
        try:
            evaluation_metrics = self.model.evaluate(
                test_loader, steps=n_test_steps, return_dict=True
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

    @staticmethod
    def _check_gpu_availability():
        """Check if GPUs are available."""
        if keras.config.backend() == "torch":
            # torch backend not yet available in keras.distribution
            import torch

            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("No GPUs available, falling back to CPU.")
                return torch.device("cpu")

        elif keras.config.backend() == "tensorflow":
            devices = keras.distribution.list_devices("gpu")
            if not devices:
                logger.warning("No GPUs available, falling back to CPU.")
                devices = keras.distribution.list_devices("cpu")
            return devices

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
        return f"Crested(data={self.data is not None}, model={self.model is not None}, config={self.config is not None})"
