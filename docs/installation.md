# Installation

You need to have Python 3.9 or newer installed on your system and a deep learning backend to be able to use CREsted.

CREsted is build on top of keras 3.0 and can therefore be used with your deep learning backend of choice (Tensorflow or Pytorch).

1. Install either [Tensorflow](https://www.tensorflow.org/install) or [Pytorch](https://pytorch.org/get-started/locally/) for GPU.
   Refer to the installation instructions on those pages to ensure you have the correct version of CUDA and cuDNN installed.
   If you don't have a preference and don't know which backend to choose, refer to [choosing your backend](#choosing-your-backend).
   If you have all the latest drivers installed, this installation boils down to doing:

```bash
pip install tensorflow[and-cuda]
# or
pip install torch
```

2. Install the latest release of `crested` from [PyPI](https://pypi.org/project/CREsted/)

```bash
pip install crested
```

3. If you plan on doing motif analysis using tf-modisco (lite) inside CREsted, you will need to run the following additional install:

```bash
pip install "modisco-lite>=2.2.1"
```

Modiscolite may require a cmake installation on your system. If you don't have it, you can install it with:

```bash
pip install cmake
```

## Choosing your backend

CREsted is build on top of keras 3.0 and can therefore be used with your deep learning backend of choice (Tensorflow or Pytorch). If you don't have a preference, you can take the following into account:

1. From our (and Keras' official) benchmarking, **Tensorflow** is generally faster than pytorch for training (up to 2x) since Tensorflow operates in graph mode whereas Pytorch uses eager mode. If you plan on training many models, Tensorflow might be the better choice.
2. **Pytorch** is easier to debug and get going. Tensorflow will easily throw a bunch of warnings or fail to detect CUDA if you don't have the exact right versions of CUDA and cuDNN installed. Pytorch seems more lenient in this regard. If you only plan on running predictions or training a few models, Pytorch might be the easier choice.
3. Current Keras 3.0 is still in active development and some features (mainly multi GPU training and weights and biases logging) are currently only supported with the Tensorflow backend. If you plan on using these features, you should choose Tensorflow. We will implement these features in a backend agnostic way as soon as Keras 3.0 has done so (it is on their roadmap).
