"""Setup loguru logging for the package."""

import sys

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: str | None = None):
    """
    Setup loguru logging for the package.

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
