OS=`uname`
SHELL=/bin/bash

conda_env_NAME=netcdf-scm
conda_env_PATH=$(MINICONDA_PATH)/envs/$(conda_env_NAME)
conda_env_MINIMAL_YML=$(PWD)/conda-environment-minimal.yaml
conda_env_DEV_YML=$(PWD)/conda-environment-dev.yaml

PIP_REQUIREMENTS_MINIMAL=$(PWD)/pip-requirements-minimal.txt
PIP_REQUIREMENTS_DEV=$(PWD)/pip-requirements-dev.txt

NOTEBOOKS_DIR=./notebooks
NOTEBOOKS_SANITIZE_FILE=$(NOTEBOOKS_DIR)/tests_sanitize.cfg

DOCS_DIR=$(PWD)/docs
LATEX_BUILD_DIR=$(DOCS_DIR)/build/latex
LATEX_BUILD_STATIC_DIR=$(LATEX_BUILD_DIR)/_static
LATEX_LOGO=$(DOCS_DIR)/source/_static/logo.png

FILES_TO_FORMAT_PYTHON=setup.py scripts src tests docs/source/conf.py


define activate_conda
	[ ! -f $(HOME)/.bash_profile ] || . $(HOME)/.bash_profile; \
	[ ! -f $(HOME)/.bashrc ] || . $(HOME)/.bashrc; \
	conda activate; \
	[ ! -z "`which conda`" ] || { echo 'conda not found'; exit 1; }
endef

define activate_conda_env
	$(call activate_conda,); \
	echo 'If this fails, install your environment with make conda-env'; \
	conda activate $(conda_env_NAME)
endef


.PHONY: docs
docs:
	make $(DOCS_DIR)/build/html/index.html

# Have to run build twice to get stuff in right place
$(DOCS_DIR)/build/html/index.html: $(DOCS_DIR)/source/*.py $(DOCS_DIR)/source/_templates/*.html $(DOCS_DIR)/source/*.rst src/netcdf_scm/*.py README.rst CHANGELOG.rst
	mkdir -p $(LATEX_BUILD_STATIC_DIR)
	cp $(LATEX_LOGO) $(LATEX_BUILD_STATIC_DIR)
	$(call activate_conda_env,); \
		cd $(DOCS_DIR); \
		make html

.PHONY: test-all
test-all:
	make test
	make test-notebooks

.PHONY: test
test:
	$(call activate_conda_env,); \
		pytest --cov -rfsxEX --cov-report term-missing

.PHONY: test-notebooks
test-notebooks:
	$(call activate_conda_env,); \
		pytest -rfsxEX --nbval $(NOTEBOOKS_DIR) --sanitize $(NOTEBOOKS_SANITIZE_FILE)

.PHONY: flake8
flake8:
	$(call activate_conda_env,); \
		flake8 $(FILES_TO_FORMAT_PYTHON)

.PHONY: black
black:
	@status=$$(git status --porcelain pymagicc tests); \
	if test "x$${status}" = x; then \
		$(call activate_conda_env,); \
		black --exclude _version.py --py36 $(FILES_TO_FORMAT_PYTHON); \
	else \
		echo Not trying any formatting. Working directory is dirty ... >&2; \
	fi;

.PHONY: new-release
new-release:
	@echo 'See instructions in the Releasing sub-section of the Development section of the docs'

.PHONY: release-on-conda
release-on-conda:
	@echo 'See instructions in the Releasing sub-section of the Development section of the docs'

# first time setup, follow this https://blog.jetbrains.com/pycharm/2017/05/how-to-publish-your-package-on-pypi/
# then this works
.PHONY: publish-on-testpypi
publish-on-testpypi:
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(call activate_conda_env,); \
			python setup.py sdist bdist_wheel --universal; \
			twine upload -r testpypi dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

.PHONY: publish-on-pypi
publish-on-pypi:
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(call activate_conda_env,); \
			python setup.py sdist bdist_wheel --universal; \
			twine upload dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

.PHONY: setup-versioneer
setup-versioneer:
	$(call activate_conda_env,); \
		versioneer install

.PHONY: conda-env-update
conda-env-update:
	@echo "Updating the environment requires this command"
	@echo "conda env update --name env-name --file env-file"
	@echo "You have to decide for yourself which file to update from and how "
	@echo "to ensure that the dev and minimal environments don't conflict, we "
	@echo "haven't worked out how to automate that."

.PHONY: conda-env
conda-env:
	# thanks https://stackoverflow.com/a/38609653 for the conda install from
	# file solution
	# tidy up pip install once I get expect exception pip installable
	$(call activate_conda,); \
		conda config --add channels conda-forge; \
		conda create -y -n $(conda_env_NAME); \
		conda activate $(conda_env_NAME); \
		conda install -y --file $(conda_env_MINIMAL_YML); \
		conda install -y --file $(conda_env_DEV_YML); \
		pip install --upgrade pip; \
		pip install -Ur $(PIP_REQUIREMENTS_MINIMAL); \
		pip install -e .[test,docs,deploy]

.PHONY: clean-docs
clean-docs:
	rm -rf $(DOCS_DIR)/build

.PHONY: clean
clean:
	$(call activate_conda,); \
		conda deactivate; \
		conda remove --name $(conda_env_NAME) --all -y
	make clean-docs

.PHONY: variables
variables:
	@echo PWD: $(PWD)
	@echo OS: $(OS)
	@echo SHELL: $(SHELL)

	@echo conda_env_NAME: $(conda_env_NAME)
	@echo conda_env_PATH: $(conda_env_PATH)
	@echo conda_env_MINIMAL_YML: $(conda_env_MINIMAL_YML)
	@echo conda_env_DEV_YML: $(conda_env_DEV_YML)

	@echo PIP_REQUIREMENTS_MINIMAL: $(PIP_REQUIREMENTS_MINIMAL)
	@echo PIP_REQUIREMENTS_DEV: $(PIP_REQUIREMENTS_DEV)

	@echo NOTEBOOKS_DIR: $(NOTEBOOKS_DIR)
	@echo NOTEBOOKS_SANITIZE_FILE: $(NOTEBOOKS_SANITIZE_FILE)

	@echo DOCS_DIR: $(DOCS_DIR)
	@echo LATEX_LOGO: $(LATEX_LOGO)

	@echo FILES_TO_FORMAT_PYTHON: $(FILES_TO_FORMAT_PYTHON)
