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
  - [Infrastructure](#infrastructure)
  - [Quick Start](#quick-start)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
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
make requirements  # run from terminal
```

Alternatively, you can install the dependencies using the following commands:
```bash
conda env create -f environment.yml
```

This will install the required packages and **tensorflow version 2.14**, which is compatible with CUDA 11.8. If you want to use a different version of CUDA, make sure you install the compatible version of tensorflow instead (see [here](https://www.tensorflow.org/install/source#gpu) for more information).

Don't forget to activate your environment before running any of the scripts:
```bash
conda activate deeppeak
```

### Dependencies

Tensorflow requires CUDA and cuDNN to be installed. If you have multiple versions installed on your machine, ensure that you have the correct versions of CUDA and cuDNN loaded in your environment. 
```bash
# Find available cuDNN/CUDA versions
module spider cudnn

# Load the correct versions for tensorflow
# !Run this each time you start a new terminal!
module load cuDNN/8.7.0.84-CUDA-11.8.0  # CUDA 11.8
```
If you don't load cuDNN/CUDA in your terminal before running the training scripts when you have multiple versions available, you will encounter the error: `tensorflow/tsl/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory` or `Could not load library libcudnn_cnn_infer.so.8. Error: libnvrtc.so: cannot open shared object file: No such file or director`

### Useful commands

To see a list of useful commands, run the following:
```bash
make help
```
For example, you can delete all compiled python files using: 
```bash
make clean_compiled
```

### Infrastructure

Deeppeak is optimized to work quickly on recent GPUs using limited amount of memory. 
For a model with 6M parameters and a dataset containing 500K regions, working on 8 cores with 5GB each should more than suffice to perform both data preprocessing, training, and validation.
If you have larger datasets you might need to increase the memory a little bit.

Increasing the number of GPUs will simply speed up training by a factor of the number of GPUs available.

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

All the processing and training can be run from the command line.  
If you prefer working in notebooks, you can do so but remember to not commit them to the git repository. Place them in the *notebooks/personal* folder, which is ignored by git (and remember to use relative imports/runs, e.g. from '../../deeppeak/training import ...').

### Data Preprocessing

The data preprocessing will create target vectors of shape (4, N, C) where N is the number of regions and C is the number of cells types. For each of these, you will have four scalar values representing the *max, mean, count, and logcount* ATAC of the target region.  
The region sizes considered for creating the target vectors are defined by the **target_len** parameter in your config file.  
As inputs, the model will use the preprocessed region bed file with region sizes that are defined by the **seq_len** parameter in your config file.  
Matching the regions to the genome and one hot encoding them happens online during training to save on both RAM and disk memory.  
These files will be created in *data/processed* by running the data pipeline.

As inputs for the data preprocessing pipeline, you'll need four types of files:
* Reference genome fasta file
* Consensus regions bed file
* Chrom.sizes file
* Folder containing bigwigs files per cell type (deeppeak) or topic bed files (deeptopic)

#### Linking the raw data

The easiest way to control your data pipeline is to create symbolic links from your original input data paths to the project's *data/raw* folder. If you want to work with the default parser arguments without having to provide the data paths each time throughout data preprocessing and training, you should see that they are named in the following way:
```
genome fasta == > data/raw/genome.fa
chromosome sizes ==> data/raw/chrom.sizes
consensus regions ==> data/raw/consensus_peaks.bed
bigwig folder ==> data/raw/bw/{cell_type_name}.bw OR topics folder ==> data/raw/topics/{topic_number}.bed
```

You can create the symbolic links manually yourself with the ```ln -s``` command, or you can use the following handy command which will create the symbolic links and name them correctly given certain patterns in the input data path:

```bash
# Command
make linkdata path='/path/to/your/source/file_or_folder'  # you should do this 4 times, once for each filetype

# For example, if your input path contains the pattern '.fa'
# This command will automatically create a symbolic path from that path 
# to data/raw/genome.fa

# Bigwigs example
make linkdata path='/path/to/my/bigwigs/folder/bw/'
```

#### Data pipeline

If you have set up your data in *data/raw* and named them correctly, you can easily run the full preprocessing pipeline with one command.

```bash
make data_pipeline_deeppeak  # deeppeak data pipeline
make data_pipeline_deeptopic  # deeptopic data pipeline
```

This will:
* Clean the bed files by extending start/end positions, filtering out negative & out of bounds coordinates
* Create two bed files with different region sizes, one that will be used as model inputs and one that is needed for creating the target vectors
* Match bigwigs to region bed files by running the all_ct_bigwigAverageOverBed.sh script (**Kentools will be required and loaded for this**)
* Create the target vectors

Alternatively, you can run through all the steps yourself by running the python scripts and bash scripts in the correct order. I don't recommend doing this, but if you want to do so you can follow the steps under "#datasets" in the *Makefile* and run the python scripts with the correct parser arguments in the correct order.

### Model Training

To train the model, change the necessary training configs in *user.yml*, ensure your environment is loaded correctly (see [Getting started](#getting-started)), and ensure that you have your input bed file and targets file from the data processing pipeline in the *data/processed* folder.

Important configs to change are:
- **model_architecture**: The model architecture to use for training. simple_convnet can be used as a baseline.
- **batch_size**: The batch size used for training. This will depend on the amount of memory you have available on your GPU. If you have a lot of memory, you can increase this to speed up training.
- **num_epochs**: The number of epochs to train for.
- **patience**: The number of epochs to wait before early stopping if the validation loss does not improve.
- **learning_rate**: The learning rate used for training. Decrease/increase if your model is not learning.
- **pretrained_model_path**: If you want to continue training from a pretrained model, you can specify the path to the model checkpoint here.
- **mixed_precision**: WARNING: This can cause numerical instability and NaNs in the loss. If you have a recent GPU you can leave this on True to speed up training. Set to False if you encounter NaNs in the loss.

Afterwards, run the following command:
```bash
make train_deeppeak  # deeppeak shortcut with default paths
make train_deeptopic  # deeptopic shortcut with default paths

# or
python enhancerai/training/train.py --genome_fasta_file /path/to/genome.fa --bed_file /path/to/consensus_peaks_inputs.bed --targets_file /path/to/targets_{task}.npz --output_dir /path/to/checkpoints_dir/
```

This will output all model checkpoints during training to your *output_dir/{project_name}/{timestamp}*, as well as save the required data files to the output directory to ensure the training is fully reproducible. Of course, this means that you will have a lot of duplicate files saved, so don't forget to **remove any checkpoint directories of runs you don't plan on keeping!**

Renaming interesting runs to something more informative than the timestamp is also a good idea.

An example slurm script to perform model training (on one GPU) would look like this:

```bash
#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=30G
#SBATCH --time=08:00:00
#SBATCH -A lp_big_wice_gpu
#SBATCH -p dedicated_big_gpu
#SBATCH -G 1
#SBATCH --cluster=wice
#SBATCH --mail-type=ALL
#SBATCH --mail-user=my.user@my.email

##TRAINING MODEL
workdir={/path/to/your/Deeppeak/repository/}
cd $workdir

source /lustre1/project/stg_00002/mambaforge/{your_user}/etc/profile.d/conda.sh
conda activate deeppeak

module load cuDNN/8.7.0.84-CUDA-11.8.0

make train_deeppeak
```

You can store your personal slurm scripts under *scripts/personal*, which is ignored by git.

### Evaluation

#### Evaluate on test set

To evaluate your trained models performance on the test set, you can run the following command:

```bash
# Without checkpoint specification, will load the last epoch from checkpoints
python deeppeak/evaluate/evaluate.py --model_dir checkpoints/{project_name}/{model_timestamp}

# Specifying model epoch
python deeppeak/evaluate/evalute.py --model_dir checkpoints/{project_name}/{model_timestamp} --model_name {epoch}.keras
```

This will output and save overall performance metrics and performance metrics per cell types for the test set, as well as save some useful plots to your *{model_dir}/evaluation_outputs/*

#### Gradient x input interpretation

To get integrated gradients explainers (very similar to shap.deepexplainer), run the `python deeppeak/evaluate/interpret.py` script with the required parser arguments. Outputs will be saved to the *{model_dir}/evaluation_outputs* (scores as a numpy file, plots as .jpeg).
Supported methods for now are 'integrated gradients' and 'expected integrated gradients'. Integrated gradients is much faster, but expected integrated gradients is more accurate since it uses more baselines.

Some examples: 

```bash
# Calculate and save gradientxinput score across entire dataset.
python deeppeak/evaluate/interpret.py --model_dir checkpoints/{project_name}/{timestamp}/

# Specify chromosome and regions and save the plots of the results (can be a wide range of regions).
# Plotting will take a long time.
python deeppeak/evaluate/interpret.py --model_dir checkpoints/{project_name}/{timestamp}/ --chromosome chr1 --region_start 3206136 --region_end 3382988 --visualize

# Use expected integrated gradients instead of integrated gradients
python deeppeak/evaluate/interpret.py --model_dir checkpoints/{project_name}/{timestamp}/--method integrated_grad

# Focus on specific cell types only
python deeppeak/evaluate/interpret.py --model_dir checkpoints/{project_name}/{timestamp}/ --class_index "[0,1,2]"

# Zoom in on center number of bases in input region and visualize
python deeppeak/evaluate/interpret.py --model_dir checkpoints/{project_name}/{timestamp}/ --zoom_n_bases 500 --visualize

# Plot fire enhancer
python deeppeak/evaluate/interpret.py --model_dir checkpoints/{project_name}/{timestamp}/ --chromosome chr18 --region_start 61107570 --region_end 61109684 --zoom_n_bases 500 --class_index "[10]" --visualize
```

**WARNING:** calculating these explainers can take a long time for many regions (IGs should be quite a bit faster than shap.deepexplainer though), and plotting them will take an even longer time. Make sure to only focus on the regions of interest.

## To Do
* FastISM
* Evaluation notebook for inspecting individual samples

## Acknowledgments
Authors:
  - Niklas Kemp (Niklas.Kempynck@kuleuven.be)  

Contributors:
  - Lukas Mahieu (Lukas.Mahieu@kuleuven.be)
  - Vasileios Konstantakos (Vasileios.Konstantakos@kuleuven.be)
  
