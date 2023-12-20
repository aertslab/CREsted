# DeepPeak: Predicting Chromatin Accessibility from Genome Sequences

![DeepPeak Logo](figures/banner.png)


DeepPeak is a sequence based deep learning model designed to predict scATAC peak height over multiple cell types. 
The model architecture is based on ChromBPNet, but, instead of predicting the entire track signal, DeepPeak predicts the average peak height.
The goal of the model is to learn to denoise the signal and to be able to interpret the results post-hoc to identify the most important regions and motifs for each prediction.

## Table of Contents
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Useful commands](#useful-commands)
  - [Quick Start](#quick-start)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Prediction](#prediction)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Getting Started
To get started with DeepPeak, clone the repository, install the dependencies, and follow the quick start guide.
The repository has been tested on Rocky Linux 8.8, python 3.10, and CUDA 11.8.

### Installation
To install the dependencies, make sure you have conda installed and run the following commands:
```bash
# Creates a new conda environment if it does not exist yet
# Updates the environment if it does exist
make requirements
```

Alternatively, you can install the dependencies using the following commands:
```bash
conda env create -f environment.yml
```

This will install the required packages and **tensorflow version 2.14**, which is compatible with CUDA 11.8. If you want to use a different version of CUDA, make sure you install the compatible version of tensorflow instead (see [here](https://www.tensorflow.org/install/source#gpu) for more information).

### Dependencies

Tensorflow requires CUDA and cuDNN to be installed. If you have multiple versions installed on your machine, ensure that you have the correct versions of CUDA and cuDNN loaded in your environment. 
```bash
# Find available cuDNN/CUDA versions
module spider cudnn

# Load the correct versions for tensorflow
module load cuDNN/8.7.0.84-CUDA-11.8.0  # CUDA 11.8
```

### Useful commands

To see a list of useful commands, run the following:
```bash
make help
```
For example, you can delete all compiled python files using: 
```bash
make clean_compiled
```

### Quick Start
In progress...

## Usage
To be able to run the preprocessing, model training, or evaluation code, you need to ensure your configs are set.  
Copy the config file from *configs/default.yml* to *configs/user.yml* (you can do so easily with the command `make copyconfig`) and make the necessary changes in the *user.yml* file.  
**Do not change the default.yml file!**

The most important changes  you need to make in your *user.yml* for each project are under the *general* section:
- **project_name**: The name of the project. This will be used to name the output files and wandb project.
- **num_classes**: The number of cell types in the dataset.
- **seq_len**: The length you want your input sequences to be (in bp). We recommend to have this wider than your target region, so that the model can also learn patterns outside of the center of the target region.
- **target_len**: The length your want your target region to be (in bp).


### Data Preprocessing
In progress...

### Model Training

To train the model, change the necessary training configs in *user.yml*, ensure your environment is loaded correctly (see [Getting started](#getting-started)) and run the following command:
```bash
make train  # shortcut with default paths

# or
python deeppeak/training/train.py --genome_fasta_file /path/to/genome.fa --bed_file /path/to/consensus_peaks_inputs.bed --targets_file /path/to/targets.npy --output_dir /path/to/output
```

### Prediction
In progress...


## Results
In progress...

## Contributing
In progress

### To Do
* ISM
* Prediction (& plots)
* Normalization checking

## Acknowledgments
Authors:
  - Niklas Kemp (Niklas.Kemp@kuleuven.be)  

Contributors:
  - Lukas Mahieu (Lukas.Mahieu@kuleuven.be)
  - Vasileios Konstantakos (Vasileios.Konstantakos@kuleuven.be)
  
