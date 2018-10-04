OS=`uname`
SHELL=/bin/bash

CONDA_ENV_NAME=netcdf-scm
CONDA_ENV_PATH=$(MINICONDA_PATH)/envs/$(CONDA_ENV_NAME)
CONDA_ENV_YML=$(PWD)/environment-minimal.yaml
CONDA_ENV_DEV_YML=$(PWD)/environment-dev.yaml

PIP_DEV_REQUIREMENTS=dev-pip-requirements.txt

define activate_conda
	[ ! -f $(HOME)/.bash_profile ] || . $(HOME)/.bash_profile; \
	[ ! -f $(HOME)/.bashrc ] || . $(HOME)/.bashrc; \
	conda activate; \
	[ ! -z "`which conda`" ] || { echo 'conda not found'; exit 1; }
endef

define activate_conda_env
	$(call activate_conda,); \
	echo 'If this fails, install your environment with make conda_env'; \
	conda activate $(CONDA_ENV_NAME)
endef

.PHONY: test
test:
	$(call activate_conda_env,); \
		pytest

.PHONY: new_release
new_release:
	@echo 'For a new release on PyPI:'
	@echo 'git tag vX.Y.Z'
	@echo 'python setup.py register sdist upload'

.PHONY: setup_versioneer
setup_versioneer:
	$(call activate_conda_env,); \
		versioneer install

.PHONY: conda_env_update
conda_env_update:
	@echo "Updating the environment requires this command"
	@echo "conda env update --name env-name --file env-file"
	@echo "You have to decide for yourself which file to update from and how "
	@echo "to ensure that the dev and minimal environments don't conflict, we "
	@echo "haven't worked out how to automate that."

.PHONY: conda_env
conda_env:
	# thanks https://stackoverflow.com/a/38609653 for the conda install from
	# dev file solution
	$(call activate_conda,); \
		conda env create -n $(CONDA_ENV_NAME) -f $(CONDA_ENV_YML); \
		conda activate $(CONDA_ENV_NAME); \
		pip install --upgrade pip; \
		while read requirement; do conda install --yes $requirement; done < $(CONDA_ENV_DEV_YML); \
		pip install -Ur $(PIP_DEV_REQUIREMENTS); \
		pip install -e .

.PHONY: variables
variables:
	@echo PWD: $(PWD)
	@echo OS: $(OS)
	@echo SHELL: $(SHELL)

	@echo CONDA_ENV_NAME: $(CONDA_ENV_NAME)
	@echo CONDA_ENV_PATH: $(CONDA_ENV_PATH)
	@echo CONDA_ENV_YML: $(CONDA_ENV_YML)

	@echo PIP_DEV_REQUIREMENTS: $(PIP_DEV_REQUIREMENTS)
