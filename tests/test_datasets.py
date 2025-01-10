"""Dataset tests."""

from __future__ import annotations

import pytest
import requests

from crested._datasets import _get_dataset_index, get_model


def test_get_model_invalid_model():
    """
    Test that get_model raises an AssertionError for an unknown model name.
    """
    with pytest.raises(AssertionError, match="not recognised"):
        get_model("not_a_valid_model")


def test_all_data_http_responses():
    """
    Test whether all available data in the registry return a valid HTTP response.
    """
    dataset_index = _get_dataset_index()
    registry = {key: dataset_index.get_url(key) for key in dataset_index.registry}
    for key, url in registry.items():
        response = requests.get(url, stream=True)
        assert response.status_code == 200, f"{key} URL {url} did not return 200 OK"

    assert len(registry) > 0, "No data found in the registry"
