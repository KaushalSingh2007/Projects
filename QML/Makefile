# Q-GUARD developer workflows.
# On Windows, run these under Git Bash, or invoke the underlying commands
# directly (see README.md "Quickstart").

.PHONY: setup data test lint format run reproduce figures clean

setup:                ## Create the conda environment and install the package (editable).
	conda env create -f environment.yml || conda env update -f environment.yml

test:                 ## Run the test suite.
	pytest

lint:                 ## Static checks.
	ruff check src tests

format:               ## Auto-format.
	black src tests
	ruff check --fix src tests

data:                 ## Download the Z24 dataset (M1).
	python -m qguard.data.download_z24

run:                  ## Run one experiment: make run EXP=E02_qguard
	python -m qguard.tracking.run experiment=$(EXP)

reproduce:            ## Run the full experiment suite end to end.
	python -m qguard.tracking.run --multirun experiment=E01_baselines,E02_qguard,E04_environment

figures:              ## Regenerate report figures from logged runs.
	python -m qguard.viz.plots

clean:                ## Remove caches and build artifacts.
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
