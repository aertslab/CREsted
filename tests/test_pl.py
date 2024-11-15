import inspect

import pytest

import crested.pl


def test_plot_functions_use_render_plot():
    # Get all submodules in crested.pl
    submodules = [crested.pl]
    for _, obj in inspect.getmembers(crested.pl):
        if inspect.ismodule(obj):
            submodules.append(obj)

    # Check each function in the submodules
    for submodule in submodules:
        for name, func in inspect.getmembers(submodule, inspect.isfunction):
            # Skip private functions
            if name.startswith("_"):
                continue

            # Get the source code of the function
            source = inspect.getsource(func)

            # Check if render_plot is used in the function
            assert (
                "render_plot" in source
            ), f"Function {name} in {submodule.__name__} does not use render_plot"


if __name__ == "__main__":
    pytest.main()
