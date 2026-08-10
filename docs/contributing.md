# Contributing guide

Scanpy provides extensive [developer documentation][scanpy developer guide], most of which applies to this project, too.
This document will not reproduce the entire content from there. Instead, it aims at summarizing the most important
information to get you started on contributing.

We assume that you are already familiar with git and with making pull requests on GitHub. If not, please refer
to the [scanpy developer guide][].


## Installing dev dependencies

In addition to the packages needed to _use_ this package, you need additional python packages to _run tests_ and _build
the documentation_.

The package is set up to use [`hatch`][hatch] as a project manager, which will manage separate virtual environments for development, testing, and documentation to avoid dependency conflicts.

### With Hatch (Recommended)

:::::{tab-set}
::::{tab-item} Hatch
:sync: hatch

On the command line, you typically interact with hatch through its command line interface (CLI).
Running one of the following commands will automatically resolve the environments for testing and
building the documentation in the background:

```bash
hatch test  # defined in the table [tool.hatch.envs.hatch-test] in pyproject.toml
hatch run docs:build  # defined in the table [tool.hatch.envs.docs]
```

### VS Code

If you are using VS code, install the [hatch-code][] extension.
Additionally, make sure that the `vscode-python-environments` extension is installed (should be by default)
and `"python.useEnvironmentsExtension": true` is activated in your `settings.json`.

Next, open the "Python Environment Managers" sidebar.
You can do so by opening the command palette (Ctrl+Shift+P) and searching for `Python: Focus on Environment Managers View`.
It will show a collapsible list where you can expand "Hatch"
and activate an environment by clicking on the checkmark next to it.
As the main development environment, we recommend to use `hatch-test` with the latest supported Python version.

### Other IDEs

For other IDEs, you’ll have to point the editor at the paths to the virtual environments manually.
To get a list of all environments for your projects, run

```bash
hatch env show -i
```

This will list “Standalone” environments and a table of “Matrix” environments like the following:

```
+------------+---------+--------------------------+----------+---------------------------------+-------------+
| Name       | Type    | Envs                     | Features | Dependencies                    | Scripts     |
+------------+---------+--------------------------+----------+---------------------------------+-------------+
| hatch-test | virtual | hatch-test.py3.12-stable | dev      | coverage-enable-subprocess==1.0 | cov-combine |
|            |         | hatch-test.py3.14-stable | test     | coverage[toml]~=7.4             | cov-report  |
|            |         | hatch-test.py3.14-pre    |          | pytest-mock~=3.12               | run         |
|            |         |                          |          | pytest-randomly~=3.15           | run-cov     |
|            |         |                          |          | pytest-rerunfailures~=14.0      |             |
|            |         |                          |          | pytest-xdist[psutil]~=3.5       |             |
|            |         |                          |          | pytest~=8.1                     |             |
+------------+---------+--------------------------+----------+---------------------------------+-------------+
```

From the `Envs` column, select the environment name you want to use for development.
As the main development environment, we recommend to use `hatch-test` with the latest supported Python version.
In this example, it would be `hatch-test.py3.14-stable`.

Next, create the environment with

```bash
hatch env create hatch-test.py3.14-stable
```

Then, obtain the path to the environment using

```bash
hatch env find hatch-test.py3.14-stable
```

and manually point it to the python binary.


::::

::::{tab-item} uv
:sync: uv

A popular choice for managing virtual environments is [uv][].
The main disadvantage compared to hatch is that it supports only a single environment per project at a time,
which requires you to mix the dependencies for running tests and building docs.
This can have undesired side-effects,
such as requiring to install a lower version of a library your project depends on,
only because an outdated sphinx plugin pins an older version.

To initialize a virtual environment in the `.venv` directory of your project, simply run

```bash
uv sync --all-extras
```

The `.venv` directory is typically automatically discovered by IDEs such as VS Code.

::::


### With (uv) pip
If installing packages with `pip`, we recommend using `uv`'s solver with `uv pip`.

```bash
cd CREsted
pip install -e ".[dev,test,doc]"
```

[hatch]: https://hatch.pypa.io/
[uv]: https://docs.astral.sh/uv/

## Code-style

This package uses [pre-commit][]-style hooks to enforce consistent code-styles.
We recommend running them with [prek][], a fast, drop-in replacement for `pre-commit` that reads the same `.pre-commit-config.yaml`.
On every commit, the checks will either automatically fix issues with the code, or raise an error message.

To enable the checks locally, install [prek][] (e.g. with `uv tool install prek`) and run

```bash
prek install
```

in the root of the repository.
prek will automatically download all dependencies when it is run for the first time.

Alternatively, you can rely on the [pre-commit.ci][] service enabled on GitHub.
If you didn’t run the checks before pushing changes to GitHub it will automatically commit fixes to your pull request, or show an error message.

If pre-commit.ci added a commit on a branch you still have been working on locally, simply use

```bash
git pull --rebase
```

to integrate the changes into yours.
While the [pre-commit.ci][] is useful, we strongly encourage installing and running the checks locally first to understand their usage.

Finally, most editors have an _autoformat on save_ feature.
Consider enabling this option for [ruff][ruff-editors] and [biome][biome-editors].

[pre-commit]: https://pre-commit.com/
[prek]: https://prek.j178.dev/
[pre-commit.ci]: https://pre-commit.ci/
[ruff-editors]: https://docs.astral.sh/ruff/integrations/
[biome-editors]: https://biomejs.dev/guides/integrate-in-editor/

(writing-tests)=

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

Continuous integration via GitHub actions will automatically run the tests on all pull requests and test
against the minimum and maximum supported Python version.

Additionally, there’s a CI job that tests against pre-releases of all dependencies (if there are any).
The purpose of this check is to detect incompatibilities of new package versions early on and
gives you time to fix the issue or reach out to the developers of the dependency before the package
is released to a wider audience.

The CI job is defined in `.github/workflows/test.yaml`,
however the single point of truth for CI jobs is the Hatch test matrix defined in `pyproject.toml`.
This means that local testing via hatch and remote testing on CI tests against the same python versions and uses the same environments.

## Publishing a release

### Updating the version number

Before making a release, you need to update the version number in the `pyproject.toml` file.
Please adhere to [Semantic Versioning][semver], in brief

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> 1. MAJOR version when you make incompatible API changes,
> 2. MINOR version when you add functionality in a backwards compatible manner, and
> 3. PATCH version when you make backwards compatible bug fixes.
>
> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

Once you are done, commit and push your changes and navigate to the "Releases" page of this project on GitHub.
Specify `vX.X.X` as a tag name and create a release.
For more information, see [managing GitHub releases][].
This will automatically create a git tag and trigger a Github workflow that creates a release on [PyPI][].

[semver]: https://semver.org/
[managing GitHub releases]: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
[pypi]: https://pypi.org/

## Writing documentation

Please write documentation for new or changed features and use-cases. This project uses [sphinx][] with the following features:

-   the [myst][] extension allows to write documentation in markdown/Markedly Structured Text
-   [Numpy-style docstrings][numpydoc] (through the [napoleon][numpydoc-napoleon] extension).
-   Jupyter notebooks as tutorials through [myst-nb][] (See [Tutorials with myst-nb](#tutorials-with-myst-nb-and-jupyter-notebooks))
-   [Sphinx autodoc typehints][], to automatically reference annotated input and output types
-   Citations (like {cite:p}`Virshup_2023`) can be included with [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/)

See the [scanpy developer docs](https://scanpy.readthedocs.io/en/latest/dev/documentation.html) for more information
on how to write documentation.

### Including new functions in the documentation (CREsted-specific)

All functions'/objects' documentation is automatically generated by `[sphinx.ext.autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)` and `[sphinx.ext.autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)` from the docstring.
We automatically generate the docs for everything in `pp`, `pl`, `tl` and `utils` recursively; that means that just writing the docstring is enough to have it show up on the readthedocs.
However, top-level functions, like `crested.import_bigwigs`, do need to be listed in `autosummary` lists manually; see `docs/api/datasets.md` for an example. (Sub)modules are documented by their docstring in `__init__.py`.

These recursive pages are created using the `docs/_templates/custom-module-template.rst`, based on Jinja. See the [Jinja template tutorial](https://jinja.palletsprojects.com/en/stable/templates/) for an introduction. The template itself is derived from [here](https://stackoverflow.com/a/62613202)/[here](https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion), but with changes for our package. The most notable is that we strip `crested.` from all titles for brevity and hard-code certain prefixes for submodules, to e.g. show 'Preprocessing: `pp`' rather than just '`pp`'.

### Tutorials with myst-nb and jupyter notebooks

The documentation is set-up to render jupyter notebooks stored in the `docs/notebooks` directory using [myst-nb][].
Currently, only notebooks in `.ipynb` format are supported that will be included with both their input and output cells.
It is your responsibility to update and re-run the notebook whenever necessary.

If you are interested in automatically running notebooks as part of the continuous integration,
please check out [this feature request][issue-render-notebooks] in the `cookiecutter-scverse` repository.

[issue-render-notebooks]: https://github.com/scverse/cookiecutter-scverse/issues/40

#### Hints

- If you refer to objects from other packages, please add an entry to `intersphinx_mapping` in `docs/conf.py`.
  Only if you do so can sphinx automatically create a link to the external documentation.
- If building the documentation fails because of a missing link that is outside your control,
  you can add an entry to the `nitpick_ignore` list in `docs/conf.py`

  (docs-building)=

### Building the docs locally

:::::{tab-set}
::::{tab-item} Hatch
:sync: hatch

```bash
hatch run docs:build
hatch run docs:open
```

::::

::::{tab-item} uv
:sync: uv

```bash
cd docs
uv run sphinx-build -M html . _build -W
(xdg-)open _build/html/index.html
```

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
