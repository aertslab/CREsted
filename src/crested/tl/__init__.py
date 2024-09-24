from importlib.util import find_spec

from loguru import logger

from . import data, losses, metrics, zoo
from ._configs import TaskConfig, default_configs
from ._crested import Crested


def _optional_function_warning(*args, **kwargs):
    logger.error(
        "The requested functionality requires the 'tfmodisco' package, which is not installed. "
        "Please install it with `pip install crested[tfmodisco]`.",
    )


if find_spec("modiscolite") is not None:
    MODISCOLITE_AVAILABLE = True
else:
    MODISCOLITE_AVAILABLE = False

if MODISCOLITE_AVAILABLE:
    try:
        import modiscolite

        from crested.tl._tfmodisco import (
            calculate_similarity_matrix,
            create_pattern_matrix,
            generate_nucleotide_sequences,
            match_h5_files_to_classes,
            pattern_similarity,
            process_patterns,
            generate_html_paths,
            calculate_mean_expression_per_cell_type,
            read_motif_to_tf_file,
            tfmodisco,
            find_pattern_matches,
            create_pattern_tf_dict,
            create_tf_ct_matrix
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise
else:
    create_pattern_matrix = _optional_function_warning
    generate_nucleotide_sequences = _optional_function_warning
    match_h5_files_to_classes = _optional_function_warning
    process_patterns = _optional_function_warning
    tfmodisco = _optional_function_warning
    calculate_similarity_matrix = _optional_function_warning
    pattern_similarity = _optional_function_warning
    calculate_mean_expression_per_cell_type = _optional_function_warning
    generate_html_paths = _optional_function_warning
    find_pattern_matches = _optional_function_warning
    read_motif_to_tf_file = _optional_function_warning
    create_pattern_tf_dict = _optional_function_warning
    create_tf_ct_matrix = _optional_function_warning


__all__ = [
    "data",
    "losses",
    "metrics",
    "zoo",
    "TaskConfig",
    "default_configs",
    "Crested",
]

if MODISCOLITE_AVAILABLE:
    __all__.extend(
        [
            "calculate_similarity_matrix",
            "create_pattern_matrix",
            "generate_nucleotide_sequences",
            "match_h5_files_to_classes",
            "pattern_similarity",
            "process_patterns",
            "tfmodisco",
            "calculate_mean_expression_per_cell_type",
            "generate_html_paths",
            "find_pattern_matches",
            "read_motif_to_tf_file",
            "create_pattern_tf_dict",
            "create_tf_ct_matrix"
        ]
    )
