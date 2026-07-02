"""Dataset tests."""

from __future__ import annotations

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from crested._datasets import _get_dataset_index, get_model


def test_get_model_invalid_model():
    """
    Test that get_model raises an AssertionError for an unknown model name.
    """
    with pytest.raises(AssertionError, match="not recognised"):
        get_model("not_a_valid_model")


def test_all_data_http_responses():
    """
    Test whether all available data (datasets and models) in the registry return a valid HTTP response.

    Uses a single keep-alive session with retries/backoff and lightweight HEAD
    requests, to avoid overloading the resources server with many short-lived
    connections (which can otherwise lead to spurious connection timeouts).
    """
    dataset_index = _get_dataset_index()
    registry = {key: dataset_index.get_url(key) for key in dataset_index.registry}

    retries = Retry(
        total=5,
        connect=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"HEAD", "GET"}),
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        for key, url in registry.items():
            response = session.head(url, allow_redirects=True, timeout=30)
            # Some servers don't allow HEAD; fall back to a streamed GET.
            if response.status_code >= 400:
                response = session.get(url, stream=True, timeout=30)
                response.close()
            assert response.status_code == 200, (
                f"{key} URL {url} did not return 200 OK (got {response.status_code})"
            )
    finally:
        session.close()

    assert len(registry) > 0, "No data found in the registry"
