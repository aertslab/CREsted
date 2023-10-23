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

## Sample data from a file that can be readable in text format and save to local
## Useful for running tests on local
sample:
	@python src/data/sample.py