"""Setup loguru logging for the package."""

from __future__ import annotations

import sys
from functools import wraps

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: str | None = None):
    """
    Setup loguru logging for the package.

    Default is set to log_level="INFO" and log_file=None each time the package is imported.

    Parameters
    ----------
    log_level : str
        Logging level. Default is "INFO".
    log_file : str
        Path to the log file. If None, logs will be printed to stdout.
    """
    logger.remove()
    logger.add(sys.stdout, level=log_level, format="{time} {level} {message}")
    if log_file:
        logger.add(
            log_file,
            level=log_level,
            format="{time} {level} {message}",
            rotation="10 MB",
        )


def log_and_raise(exception_class: Exception):
    """Decorator to both log and raise exceptions."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_class as e:
                logger.exception(e)
                raise

        return wrapper

    return decorator
