"""Backend setup utilities for Keras."""

import os
from importlib.util import find_spec


def setup_backend():
    """
    Set up Keras backend.

    Called automatically when importing modules that require keras (e.g., crested.tl).
    Checks for available backends and sets KERAS_BACKEND environment variable.
    """
    if "KERAS_BACKEND" in os.environ:
        return  # Already set by user or previous call

    # Check which backend is available without importing
    if find_spec("tensorflow") is not None:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["AUTOGRAPH_VERBOSITY"] = "0"
    elif find_spec("torch") is not None:
        os.environ["KERAS_BACKEND"] = "torch"
    else:
        raise ImportError(
            "No backend found. Please install either tensorflow or pytorch to use "
            "modules that require keras (e.g., crested.tl)."
        )
