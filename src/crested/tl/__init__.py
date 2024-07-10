from loguru import logger

from . import data, losses, metrics, zoo
from ._configs import TaskConfig, default_configs
from ._crested import Crested


def _optional_function_warning(*args, **kwargs):
    logger.error(
        "The requested functionality requires the 'tfmodisco' package, which is not installed. "
        "Please install it with `pip install crested[tfmodisco]`.",
    )


# Conditional import for tfmodisco since optional
try:
    from ._tfmodisco import (
        create_pattern_matrix,
        generate_nucleotide_sequences,
        match_h5_files_to_classes,
        process_patterns,
        tfmodisco,
    )
except ImportError:
    create_pattern_matrix = _optional_function_warning
    generate_nucleotide_sequences = _optional_function_warning
    match_h5_files_to_classes = _optional_function_warning
    process_patterns = _optional_function_warning
    tfmodisco = _optional_function_warning

if tfmodisco is not None:
    __all__ = [
        "data",
        "losses",
        "metrics",
        "zoo",
        "TaskConfig",
        "default_configs",
        "Crested",
        "create_pattern_matrix",
        "generate_nucleotide_sequences",
        "match_h5_files_to_classes",
        "process_patterns",
        "tfmodisco",
    ]
else:
    __all__ = [
        "data",
        "losses",
        "metrics",
        "zoo",
        "TaskConfig",
        "default_configs",
        "Crested",
    ]
