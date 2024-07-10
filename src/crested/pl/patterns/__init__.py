from loguru import logger

from ._contribution_scores import contribution_scores


def _optional_function_warning(*args, **kwargs):
    logger.error(
        "The requested functionality requires the 'tfmodisco' package, which is not installed. "
        "Please install it with `pip install crested[tfmodisco]`.",
    )


try:
    from ._modisco_results import create_clustermap, modisco_results
except ImportError:
    modisco_results = _optional_function_warning
    create_clustermap = _optional_function_warning

if modisco_results is not None:
    __all__ = ["contribution_scores", "modisco_results", "create_clustermap"]
else:
    __all__ = ["contribution_scores"]
