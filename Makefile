OS=`uname`
SHELL=/bin/bash

CONDA_ENV_NAME=netcdf-scm
CONDA_ENV_PATH=$(MINICONDA_PATH)/envs/$(CONDA_ENV_NAME)
CONDA_ENV_YML=$(PWD)/environment-dev.yaml

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
	$(call activate_conda_env,); \
		conda deactivate; \
		conda env update --name $(CONDA_ENV_NAME) --file $(CONDA_ENV_YML); \
		$(call activate_conda_env,)

.PHONY: conda_env
conda_env:
	$(call activate_conda,); \
		conda env create -f $(CONDA_ENV_YML); \
		pip install -e .

.PHONY: variables
variables:
	@echo PWD: $(PWD)
	@echo OS: $(OS)
	@echo SHELL: $(SHELL)

	@echo CONDA_ENV_NAME: $(CONDA_ENV_NAME)
	@echo CONDA_ENV_PATH: $(CONDA_ENV_PATH)
	@echo CONDA_ENV_YML: $(CONDA_ENV_YML)
