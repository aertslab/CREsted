.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

ifeq (,$(shell which conda))
$(error "Error: Conda must be installed")
endif

ENV_NAME=deeppeak

ifeq (,$(shell conda env list | grep $(ENV_NAME)))
HAS_ENV=False
else
HAS_ENV=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install or update requirements
requirements:
	@if [ "$(HAS_ENV)" = "False" ]; then \
		echo "Installing conda environment: $(ENV_NAME)"; \
		conda env create --file environment.yml; \
	else \
		echo "Updating conda environment: $(ENV_NAME)"; \
		/bin/bash -c "source activate $(ENV_NAME)"; \
		conda env update --file environment.yml --prune; \
	fi

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete wandb files
clean_wandb:
	rm -rf wandb

## Delete all slurm logs
clean_logs:
	find . -type f -name "*slurm*" -delete

## Lint using flake8 on src/ while ignoring 'line too long' errors
lint:
	flake8 src/ --ignore=E501

## Link data to raw and rename (use absolute paths)
linkdata:
ifndef SRC
	$(error "SRC is not set. Use make copydata SRC=/path/to/your/source/file_or_folder")
endif
	@if [ -f $(SRC) ]; then \
		FILE_NAME=$(shell basename $(SRC)); \
		if echo $$FILE_NAME | grep -q ".bed"; then \
			echo "Creating symlink to data/raw and renaming $$FILE_NAME to consensus_peaks.bed..."; \
			ln -s $(SRC) data/raw/consensus_peaks.bed; \
		elif echo $$FILE_NAME | grep -q "chrom.sizes"; then \
			echo "Creating symlink to data/raw and renaming $$FILE_NAME to chrom.sizes..."; \
			ln -s $(SRC) data/raw/chrom.sizes; \
		elif echo $$FILE_NAME | grep -q ".fa"; then \
			echo "Creating symlink to data/raw and renaming $$FILE_NAME to genome.fa..."; \
			ln -s $(SRC) data/raw/genome.fa; \
		else \
			echo "Creating symlink for $$FILE_NAME without renaming..."; \
			ln -s $(SRC) data/raw/$$FILE_NAME; \
		fi \
	elif [ -d $(SRC) ]; then \
		BW_COUNT=$(shell find $(SRC) -maxdepth 1 -name "*.bw" | wc -l); \
		if [ "$$BW_COUNT" -eq "0" ]; then \
			echo "No .bw files found in $(SRC)."; \
		else \
			echo "Creating symlinks for .bw files in $(SRC) to data/raw/bw..."; \
			mkdir -p data/raw/bw; \
			for file in $(SRC)/*.bw; do \
				ln -s $$file data/raw/bw/; \
			done \
		fi \
	else \
		echo "$(SRC) is neither a file nor a directory!"; \
	fi

## Make Datasets
bed:
	python src/data/preprocess_bed.py data/raw data/interim

inputs: bed
	python src/data/create_inputs.py data/raw data/interim

bigwig: bed
	echo "Creating bigwig files..."
	echo "Warning: removing data/interim/bw/"
	rm -rf data/interim/bw
	scripts/all_ct_bigwigAverageOverBed.sh -o "data/interim/bw/" -b "data/raw/bw/" -p "data/interim/consensus_peaks_1000.bed"

targets: bigwig
	python src/data/create_targets.py data/interim data/interim

split:
	python src/data/train_val_test_split.py data/interim data/processed

data: inputs targets split # will run everything

## Model training
train:
	python src/models/train_model.py data/processed checkpoints/
