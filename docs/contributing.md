# Contributing guide

Scanpy provides extensive [developer documentation][scanpy developer guide], most of which applies to this project, too.
This document will not reproduce the entire content from there. Instead, it aims at summarizing the most important
information to get you started on contributing.

We assume that you are already familiar with git and with making pull requests on GitHub. If not, please refer
to the [scanpy developer guide][].

## Installing dev dependencies

In addition to the packages needed to _use_ this package, you need additional python packages to _run tests_ and _build
the documentation_.

We strongly recommend using [Hatch][] as a project manager, which will manage separate virtual environments for development, testing, and documentation to avoid dependency conflicts.

### With Hatch (Recommended)

```bash
hatch test
hatch run docs:build
```

If you need to use a specific environment with your IDE, you can find out which environment hatch uses with

```bash
hatch env show -i
```

and then create and locate it with

```bash
hatch env create hatch-test.py3.14-stable  # or the name of the environment you want to use
hatch env find hatch-test.py3.14-stable
```

### With uv

Alternatively, you can use [uv][] to set up a single environment:

```bash
uv sync --all-extras
```

### With pip

```bash
cd CREsted
pip install -e ".[dev,test,doc]"
```

[hatch]: https://hatch.pypa.io/
[uv]: https://docs.astral.sh/uv/

## Code-style

This package uses [pre-commit][] to enforce consistent code-styles.
On every commit, pre-commit checks will either automatically fix issues with the code, or raise an error message.

To enable pre-commit locally, simply run

```bash
pre-commit install
```

in the root of the repository. Pre-commit will automatically download all dependencies when it is run for the first time.

Alternatively, you can rely on the [pre-commit.ci][] service enabled on GitHub. If you didn't run `pre-commit` before
pushing changes to GitHub it will automatically commit fixes to your pull request, or show an error message.

If pre-commit.ci added a commit on a branch you still have been working on locally, simply use

```bash
git pull --rebase
```

to integrate the changes into yours.
While the [pre-commit.ci][] is useful, we strongly encourage installing and running pre-commit locally first to understand its usage.

Finally, most editors have an _autoformat on save_ feature. Consider enabling this option for [ruff][ruff-editors]
and [biome][biome-editors].

[ruff-editors]: https://docs.astral.sh/ruff/integrations/
[biome-editors]: https://biomejs.dev/guides/integrate-in-editor/

## Writing tests

This package uses the [pytest][] for automated testing. Please [write tests][scanpy-test-docs] for every function added
to the package.

Most IDEs integrate with pytest and provide a GUI to run tests. Alternatively, you can run all tests from the
command line by executing

### With Hatch (Recommended)

CREsted supports both TensorFlow and PyTorch backends. The test environment matrix tests across Python versions (3.11, 3.12, 3.13) and backends (tensorflow, pytorch):

```bash
hatch test                                      # Quick test (auto-selects compatible environment)
hatch test --all                                # Run tests on all Python versions and backends
hatch test -i backend=tensorflow                # Test only with TensorFlow backend
hatch test -i backend=pytorch                   # Test only with PyTorch backend
hatch run hatch-test.py3.11-tensorflow:run      # Specific Python + backend combination
```

### With uv

First install a backend, then run tests:

```bash
uv pip install --system tensorflow  # or torch
uv run pytest
```

### With pip

First install a backend, then run tests:

```bash
pip install tensorflow  # or torch
pytest
```

### Continuous integration

Continuous integration will automatically run the tests on all pull requests and test
against the minimum and maximum supported Python version.

Additionally, there's a CI job that tests against pre-releases of all dependencies
(if there are any). The purpose of this check is to detect incompatibilities
of new package versions early on and gives you time to fix the issue or reach
out to the developers of the dependency before the package is released to a wider audience.

[scanpy-test-docs]: https://scanpy.readthedocs.io/en/latest/dev/testing.html#writing-tests

## Publishing a release

### Updating the version number

Before making a release, you need to update the version number in the `pyproject.toml` file. Please adhere to [Semantic Versioning][semver], in brief

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1.  MAJOR version when you make incompatible API changes,
> 2.  MINOR version when you add functionality in a backwards compatible manner, and
> 3.  PATCH version when you make backwards compatible bug fixes.
>
> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

Once you are done, commit and push your changes and navigate to the "Releases" page of this project on GitHub.
Specify `vX.X.X` as a tag name and create a release. For more information, see [managing GitHub releases][]. This will automatically create a git tag and trigger a Github workflow that creates a release on PyPI.

## Writing documentation

Please write documentation for new or changed features and use-cases. This project uses [sphinx][] with the following features:

-   the [myst][] extension allows to write documentation in markdown/Markedly Structured Text
-   [Numpy-style docstrings][numpydoc] (through the [napoloen][numpydoc-napoleon] extension).
-   Jupyter notebooks as tutorials through [myst-nb][] (See [Tutorials with myst-nb](#tutorials-with-myst-nb-and-jupyter-notebooks))
-   [Sphinx autodoc typehints][], to automatically reference annotated input and output types
-   Citations (like {cite:p}`Virshup_2023`) can be included with [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/)

See the [scanpy developer docs](https://scanpy.readthedocs.io/en/latest/dev/documentation.html) for more information
on how to write documentation.

### Tutorials with myst-nb and jupyter notebooks

The documentation is set-up to render jupyter notebooks stored in the `docs/notebooks` directory using [myst-nb][].
Currently, only notebooks in `.ipynb` format are supported that will be included with both their input and output cells.
It is your responsibility to update and re-run the notebook whenever necessary.

If you are interested in automatically running notebooks as part of the continuous integration, please check
out [this feature request](https://github.com/scverse/cookiecutter-scverse/issues/40) in the `cookiecutter-scverse`
repository.

#### Hints

-   If you refer to objects from other packages, please add an entry to `intersphinx_mapping` in `docs/conf.py`. Only
    if you do so can sphinx automatically create a link to the external documentation.
-   If building the documentation fails because of a missing link that is outside your control, you can add an entry to
    the `nitpick_ignore` list in `docs/conf.py`

#### Building the docs locally

### With Hatch

```bash
hatch run docs:build
open docs/_build/html/index.html
```

### With uv or pip

```bash
cd docs
make html
open _build/html/index.html
```

<!-- Links -->

[scanpy developer guide]: https://scanpy.readthedocs.io/en/latest/dev/index.html
[cookiecutter-scverse-instance]: https://cookiecutter-scverse-instance.readthedocs.io/en/latest/template_usage.html
[github quickstart guide]: https://docs.github.com/en/get-started/quickstart/create-a-repo?tool=webui
[codecov]: https://about.codecov.io/sign-up/
[codecov docs]: https://docs.codecov.com/docs
[codecov bot]: https://docs.codecov.com/docs/team-bot
[codecov app]: https://github.com/apps/codecov
[pre-commit.ci]: https://pre-commit.ci/
[readthedocs.org]: https://readthedocs.org/
[myst-nb]: https://myst-nb.readthedocs.io/en/latest/
[jupytext]: https://jupytext.readthedocs.io/en/latest/
[pre-commit]: https://pre-commit.com/
[anndata]: https://github.com/scverse/anndata
[mudata]: https://github.com/scverse/mudata
[pytest]: https://docs.pytest.org/
[semver]: https://semver.org/
[sphinx]: https://www.sphinx-doc.org/en/master/
[myst]: https://myst-parser.readthedocs.io/en/latest/intro.html
[numpydoc-napoleon]: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[sphinx autodoc typehints]: https://github.com/tox-dev/sphinx-autodoc-typehints
[pypi]: https://pypi.org/
[managing GitHub releases]: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
