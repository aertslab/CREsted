.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

ifeq (,$(shell which conda))
$(error "Error: Conda must be installed")
endif

ENV_NAME=deeppeak

ifeq (,$(shell conda env list | grep -w $(ENV_NAME)))
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
		conda env update --file environment.yml --prune; \
	fi

## Delete all compiled Python files
clean_compiled:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

## Delete all local wandb files
clean_wandb:
	rm -rf wandb

## Delete all slurm logs
clean_logs:
	find . -type f -name "*slurm*" -delete

## Lint using flake8 on deeppeak/ while ignoring 'line too long' errors
lint:
	flake8 deeppeak/ --ignore=E501

## Link data to raw and rename (use absolute paths)
linkdata:
ifndef deeppeak
	$(error "deeppeak is not set. Use make copydata deeppeak=/path/to/your/source/file_or_folder")
endif
	@if [ -f $(deeppeak) ]; then \
		FILE_NAME=$(shell basename $(deeppeak)); \
		if echo $$FILE_NAME | grep -q ".bed"; then \
			echo "Creating symlink to data/raw and renaming $$FILE_NAME to consensus_peaks.bed..."; \
			ln -s $(deeppeak) data/raw/consensus_peaks.bed; \
		elif echo $$FILE_NAME | grep -q "chrom.sizes"; then \
			echo "Creating symlink to data/raw and renaming $$FILE_NAME to chrom.sizes..."; \
			ln -s $(deeppeak) data/raw/chrom.sizes; \
		elif echo $$FILE_NAME | grep -q ".fa"; then \
			echo "Creating symlink to data/raw and renaming $$FILE_NAME to genome.fa..."; \
			ln -s $(deeppeak) data/raw/genome.fa; \
		else \
			echo "Creating symlink for $$FILE_NAME without renaming..."; \
			ln -s $(deeppeak) data/raw/$$FILE_NAME; \
		fi \
	elif [ -d $(deeppeak) ]; then \
		BW_COUNT=$(shell find $(deeppeak) -maxdepth 1 -name "*.bw" | wc -l); \
		if [ "$$BW_COUNT" -eq "0" ]; then \
			echo "No .bw files found in $(deeppeak)."; \
		else \
			echo "Creating symlinks for .bw files in $(deeppeak) to data/raw/bw..."; \
			mkdir -p data/raw/bw; \
			for file in $(deeppeak)/*.bw; do \
				ln -s $$file data/raw/bw/; \
			done \
		fi \
	else \
		echo "$(deeppeak) is neither a file nor a directory!"; \
	fi

# Datasets
## Data: clean region bed file for inputs
data_bed_inputs:
	python deeppeak/data/preprocess_bed.py -it "inputs"
## Data: clean region bed file for targets
data_bed_targets:
	python deeppeak/data/preprocess_bed.py -it "targets" -o "data/interim"

## Data: match bigwig files to target bed file
data_bigwig:
	echo "Matching bigwig files to target bed regions..."
	echo "Warning: removing data/interim/bw/"
	rm -rf data/interim/bw
	scripts/all_ct_bigwigAverageOverBed.sh -o "data/interim/bw/" -b "data/raw/bw/" -p "data/interim/consensus_peaks_targets.bed"

## Data: create target vectors from bigwigs
data_targets:
	python deeppeak/data/create_targets.py --bigwig_dir "data/interim/bw/" --output_dir "data/processed/"

## Data: run full preprocessing pipeline
data_pipeline: data_bed_inputs data_bed_targets data_bigwig data_targets

## Model training
train:
	python deeppeak/training/train.py

## Copy configs/default.yml to configs/user.yml
copyconfig:
	cp configs/default.yml configs/user.yml


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')