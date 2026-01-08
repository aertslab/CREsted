# Installation

You need to have Python 3.11 or newer installed on your system. Installation takes around five minutes.

**Note:** CREsted's preprocessing, data import, plotting, and analysis functions (e.g., `crested.import_bigwigs()`, `crested.pp.*`) work without a deep learning backend. You only need to install TensorFlow or PyTorch if you plan to train models or use the `crested.tl` module for predictions and model-related tasks.

CREsted is built on top of Keras 3 and can therefore be used with your deep learning backend of choice (TensorFlow or PyTorch).

We recommend using [uv](https://docs.astral.sh/uv/) for package installation, which is significantly faster than pip and has better dependency resolution:

```bash
pip install uv
```

## Basic Installation (preprocessing and data import only)

If you only need data preprocessing and import functionality:

```bash
uv pip install crested
```

## Full Installation (with deep learning backend)

For training models and using the full feature set:

1. Install either [Tensorflow](https://www.tensorflow.org/install) or [Pytorch](https://pytorch.org/get-started/locally/) for GPU.
   Refer to the installation instructions on those pages to ensure you have the correct version of CUDA and cuDNN installed.
   If you don't have a preference and don't know which backend to choose, refer to [choosing your backend](#choosing-your-backend).
   If you have all the latest drivers installed, this installation boils down to doing:

```bash
uv pip install tensorflow[and-cuda]
# or
uv pip install torch
```

2. Install the latest release of `crested` from [PyPI](https://pypi.org/project/CREsted/)

```bash
uv pip install crested
```

3. If you plan on doing motif analysis using tf-modisco (lite) inside CREsted, you will need to install with the motif extra:

```bash
uv pip install "crested[motif]"
```

**Note:** TOMTOM motif matching (via memelite) is only available for Python 3.12 and earlier due to numpy compatibility constraints. Python 3.13 users can still use all other modisco features.

Modiscolite may require a cmake installation on your system. If you don't have it, you can install it with:

```bash
uv pip install cmake
```

## Choosing your backend

CREsted is build on top of keras 3.0 and can therefore be used with your deep learning backend of choice (Tensorflow or Pytorch). If you don't have a preference, you can take the following into account:

1. From our (and Keras' official) benchmarking, **Tensorflow** is generally faster than pytorch for training (up to 2x) since Tensorflow operates in graph mode whereas Pytorch uses eager mode. If you plan on training many models, Tensorflow might be the better choice.
2. **Pytorch** is easier to debug and get going. Tensorflow will easily throw a bunch of warnings or fail to detect CUDA if you don't have the exact right versions of CUDA and cuDNN installed. Pytorch seems more lenient in this regard. If you only plan on running predictions or training a few models, Pytorch might be the easier choice.
3. Current Keras 3.0 is still in active development and some features (mainly multi GPU training and weights and biases logging) are currently only supported with the Tensorflow backend. If you plan on using these features, you should choose Tensorflow. We will implement these features in a backend agnostic way as soon as Keras 3.0 has done so (it is on their roadmap).
