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
	python src/models/train.py


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