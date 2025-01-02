"""Dataset tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from crested._datasets import _get_dataset_index, get_model


@pytest.mark.parametrize("valid_model", ["model1", "model2"])
def test_get_model_valid_models(valid_model, tmp_path):
    """
    Test that get_model returns the correct paths for valid model names,
    and that it reads the output classes from the correct file.
    """
    # create mock files to not actually download the models
    model_dir = tmp_path / valid_model
    model_dir.mkdir()

    model_file = model_dir / f"{valid_model}.keras"
    model_file.touch()

    output_classes_file = tmp_path / f"{valid_model}_output_classes.tsv"
    with open(output_classes_file, "w") as f:
        f.write("classA\nclassB\nclassC\n")

    # patch `_get_dataset_index().fetch` so that it returns the path to our dummy model_dir
    with patch("crested._datasets._get_dataset_index") as mock_get_dataset_index:
        mock_index = MagicMock()
        mock_index.fetch.return_value = str(model_dir)
        mock_get_dataset_index.return_value = mock_index

        returned_model_file, returned_output_classes = get_model(valid_model)

        assert returned_model_file == str(model_file)
        assert returned_output_classes == ["classA", "classB", "classC"]


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
