.DEFAULT_GOAL := help

OS=`uname`
SHELL=/bin/bash

CONDA_ENV_YML=environment.yml
CONDA_ENV_DEV_YML=environment-dev.yml

FILES_TO_FORMAT_PYTHON=setup.py scripts src tests docs/source/conf.py

DOCS_DIR=$(PWD)/docs
LATEX_BUILD_DIR=$(DOCS_DIR)/build/latex
LATEX_BUILD_STATIC_DIR=$(LATEX_BUILD_DIR)/_static
LATEX_LOGO=$(DOCS_DIR)/source/_static/logo.png

NOTEBOOKS_DIR=./notebooks
NOTEBOOKS_SANITIZE_FILE=$(NOTEBOOKS_DIR)/tests_sanitize.cfg

ifndef CONDA_PREFIX
$(error Conda environment not active. Activate your conda environment before using this Makefile.)
else
ifeq ($(CONDA_DEFAULT_ENV),base)
$(error Do not install to conda base environment. Activate a different conda environment and rerun make. A new environment can be created with e.g. `conda create --name netcdf-scm`.))
endif
VENV_DIR=$(CONDA_PREFIX)
endif

PYTHON=$(VENV_DIR)/bin/python
COVERAGE=$(VENV_DIR)/bin/coverage

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: clean-docs
clean-docs:  ## remove all the documentation build files
	rm -rf $(DOCS_DIR)/build

.PHONY: clean-conda-env
clean-conda-env:  ## remove the conda environment
	$(CONDA_EXE) deactivate; @echo $(CONDA_DEFAULT_ENV)
	# remove --name

.PHONY: clean
clean:  ## remove the conda environment and clean the docs
	# $(call activate_conda,); \
	# 	conda deactivate; \
	# 	conda remove --name $(conda_env_NAME) --all -y
	make clean-conda-env
	make clean-docs

.PHONY: docs
docs:  ## make docs
	make $(DOCS_DIR)/build/html/index.html

# Have to run build twice to get stuff in right place
$(DOCS_DIR)/build/html/index.html: $(DOCS_DIR)/source/*.py $(DOCS_DIR)/source/_templates/*.html $(DOCS_DIR)/source/*.rst src/netcdf_scm/*.py README.rst CHANGELOG.rst $(VENV_DIR)
	mkdir -p $(LATEX_BUILD_STATIC_DIR)
	cp $(LATEX_LOGO) $(LATEX_BUILD_STATIC_DIR)
	cd $(DOCS_DIR); make html

.PHONY: flake8
flake8: $(VENV_DIR)  ## check compliance with pep8
	$(VENV_DIR)/bin/flake8 $(FILES_TO_FORMAT_PYTHON)

.PHONY: black
black: $(VENV_DIR)  ## use black to autoformat code
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/black --exclude _version.py --py36 $(FILES_TO_FORMAT_PYTHON); \
	else \
		echo Not trying any formatting, working directory is dirty... >&2; \
	fi;

.PHONY: test-all
test-all:  ## run the testsuite and test the notebooks
	make test
	make test-notebooks

.PHONY: test
test: $(VENV_DIR)  ## run the testsuite
	$(VENV_DIR)/bin/pytest --cov -rfsxEX --cov-report term-missing

.PHONY: test-notebooks
test-notebooks: $(VENV_DIR)  ## test the notebooks
	$(VENV_DIR)/bin/pytest -rfsxEX --nbval $(NOTEBOOKS_DIR) --sanitize $(NOTEBOOKS_SANITIZE_FILE)

.PHONY: new-release
new-release:  ## make a new release
	@echo 'See instructions in the Releasing sub-section of the Development section of the docs'

.PHONY: release-on-conda
release-on-conda:  ## make a new release on conda
	@echo 'See instructions in the Releasing sub-section of the Development section of the docs'

# first time setup, follow this https://blog.jetbrains.com/pycharm/2017/05/how-to-publish-your-package-on-pypi/
# then this works
.PHONY: publish-on-testpypi
publish-on-testpypi: $(VENV_DIR)  ## publish the current state of the repository to test PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py sdist bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine upload -r testpypi dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test-testpypi-install: $(VENV_DIR)  ## test whether installing from test PyPI works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	# Install dependencies not on testpypi registry
	$(TEMPVENV)/bin/pip install pandas
	# Install pymagicc without dependencies.
	$(TEMPVENV)/bin/pip install \
		-i https://testpypi.python.org/pypi netcdf-scm \
		--no-dependencies --pre
		# Remove local directory from path to get actual installed version.
	@echo "This doesn't test dependencies"
	$(TEMPVENV)/bin/python -c "import sys; sys.path.remove(''); import netcdf_scm; print(netcdf_scm.__version__)"

.PHONY: publish-on-pypi
publish-on-pypi:  $(VENV_DIR) ## publish the current state of the repository to PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py sdist bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine upload dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test-pypi-install: $(VENV_DIR)  ## test whether installing from PyPI works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install netcdf_scm --pre
	$(TEMPVENV)/bin/python scripts/test_install.py

.PHONY: test-install
test-install: $(VENV_DIR)  ## test whether installing the local setup works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install .
	$(TEMPVENV)/bin/python scripts/test_install.py

.PHONY: venv
venv:  $(VENV_DIR)  ## make virtual environment for development
$(VENV_DIR): $(CONDA_ENV_YML) $(CONDA_ENV_DEV_YML) setup.py
	$(CONDA_EXE) config --add channels conda-forge 
	$(CONDA_EXE) install -y --file $(CONDA_ENV_YML)
	$(CONDA_EXE) install -y --file $(CONDA_ENV_DEV_YML)
	# Install the remainder of the dependencies using pip
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e .[dev]
	touch $(VENV_DIR)

.PHONY: variables
variables:  ## display the value of all variables in the Makefile
	@echo CONDA_PREFIX: $(CONDA_PREFIX)
	@echo CONDA_DEFAULT_ENV: $(CONDA_DEFAULT_ENV)
	@echo CONDA_EXE: $(CONDA_EXE)
	@echo VENV_DIR: $(VENV_DIR)
	@echo PYTHON: $(PYTHON)
	@echo COVERAGE: $(COVERAGE)
	@echo ""
	@echo PWD: $(PWD)
	@echo OS: $(OS)
	@echo SHELL: $(SHELL)
	@echo ""
	@echo CONDA_ENV_YML: $(CONDA_ENV_YML)
	@echo CONDA_ENV_DEV_YML: $(CONDA_ENV_DEV_YML)
	@echo ""
	@echo PIP_REQUIREMENTS_MINIMAL: $(PIP_REQUIREMENTS_MINIMAL)
	@echo PIP_REQUIREMENTS_DEV: $(PIP_REQUIREMENTS_DEV)
	@echo ""
	@echo NOTEBOOKS_DIR: $(NOTEBOOKS_DIR)
	@echo NOTEBOOKS_SANITIZE_FILE: $(NOTEBOOKS_SANITIZE_FILE)
	@echo ""
	@echo DOCS_DIR: $(DOCS_DIR)
	@echo LATEX_LOGO: $(LATEX_LOGO)
	@echo ""
	@echo FILES_TO_FORMAT_PYTHON: $(FILES_TO_FORMAT_PYTHON)
