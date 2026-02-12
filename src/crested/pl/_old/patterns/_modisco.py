
from loguru import logger

import crested


def class_instances(*args, **kwargs):
    """
    Use :func:`~crested.pl.modisco.class_instances` instead, this alias is deprecated.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.class_instances` has been renamed to `crested.pl.modisco.class_instances` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.modisco.class_instances(*args, **kwargs)


def clustermap(*args, **kwargs):
    """
    Use :func:`~crested.pl.modisco.clustermap` instead, this alias is deprecated.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.clustermap` has been renamed to `crested.pl.modisco.clustermap` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.modisco.clustermap(*args, **kwargs)


def clustermap_tf_motif(*args, **kwargs):
    """
    Use :func:`~crested.pl.modisco.clustermap_tf_motif` instead, this alias is deprecated.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.clustermap_tf_motif` has been renamed to `crested.pl.modisco.clustermap_tf_motif` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.modisco.clustermap_tf_motif(*args, **kwargs)


def clustermap_tomtom_similarities(*args, **kwargs):
    """
    Use :func:`~crested.pl.modisco.clustermap_tomtom_similarities` instead, this alias is deprecated.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.clustermap_tomtom_similarities` has been renamed to `crested.pl.modisco.clustermap_tomtom_similarities` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.modisco.clustermap_tomtom_similarities(*args, **kwargs)


def clustermap_with_pwm_logos(*args, **kwargs):
    """
    Use :func:`~crested.pl.modisco.clustermap_with_pwm_logos` instead, this alias is deprecated.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.clustermap_with_pwm_logos` has been renamed to `crested.pl.modisco.clustermap_with_pwm_logos` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.modisco.clustermap_with_pwm_logos(*args, **kwargs)


def modisco_results(*args, **kwargs):
    """
    Use :func:`~crested.pl.modisco.modisco_results` instead, this alias is deprecated.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.modisco_results` has been renamed to `crested.pl.modisco.modisco_results` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.modisco.modisco_results(*args, **kwargs)


def selected_instances(*args, **kwargs):
    """
    Use :func:`~crested.pl.modisco.selected_instances` instead, this alias is deprecated.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.selected_instances` has been renamed to `crested.pl.modisco.selected_instances` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.modisco.selected_instances(*args, **kwargs)

def similarity_heatmap(*args, **kwargs):
    """
    Use :func:`~crested.pl.modisco.similarity_heatmap` instead, this alias is deprecated.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.similarity_heatmap` has been renamed to `crested.pl.modisco.similarity_heatmap` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.modisco.similarity_heatmap(*args, **kwargs)

def tf_expression_per_cell_type(*args, **kwargs):
    """
    Use :func:`~crested.pl.modisco.tf_expression_per_cell_type` instead, this alias is deprecated.

    :meta private:
    """
    logger.info(
        "`crested.pl.patterns.tf_expression_per_cell_type` has been renamed to `crested.pl.modisco.tf_expression_per_cell_type` in version 2.0.0"
        " and this alias will be removed in a future release. Please use its new name instead."
    )
    return crested.pl.modisco.tf_expression_per_cell_type(*args, **kwargs)
