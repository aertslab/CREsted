"""Init file for the tests module."""

import os

# Set keras backend
try:
    import tensorflow as tf

    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
except ImportError:
    try:
        import torch

        os.environ["KERAS_BACKEND"] = "torch"
    except ImportError as e:
        raise ImportError(
            "No backend found. Please install either tensorflow or pytorch."
        ) from e
