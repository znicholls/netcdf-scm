OS=`uname`
SHELL=/bin/bash

CONDA_ENV_NAME=netcdf-scm
CONDA_ENV_PATH=$(MINICONDA_PATH)/envs/$(CONDA_ENV_NAME)
CONDA_ENV_MINIMAL_YML=$(PWD)/conda-environment-minimal.yaml
CONDA_ENV_DEV_YML=$(PWD)/conda-environment-dev.yaml

PIP_REQUIREMENTS_MINIMAL=$(PWD)/pip-requirements-minimal.txt
PIP_REQUIREMENTS_DEV=$(PWD)/pip-requirements-dev.txt

NOTEBOOKS_DIR=./notebooks
NOTEBOOKS_SANITIZE_FILE=$(NOTEBOOKS_DIR)/tests_sanitize.cfg

DOCS_DIR=$(PWD)/docs
LATEX_BUILD_DIR=$(DOCS_DIR)/build/latex
LATEX_BUILD_STATIC_DIR=$(LATEX_BUILD_DIR)/_static
LATEX_LOGO=$(DOCS_DIR)/source/_static/logo.png


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


.PHONY: docs
docs:
	make $(DOCS_DIR)/build/html/index.html

# Have to run build twice to get stuff in right place
$(DOCS_DIR)/build/html/index.html: $(DOCS_DIR)/source/index.rst
	mkdir -p $(LATEX_BUILD_STATIC_DIR)
	cp $(LATEX_LOGO) $(LATEX_BUILD_STATIC_DIR)
	$(call activate_conda_env,); \
		cd $(DOCS_DIR); \
		make html

.PHONY: test_all
test_all:
	make test
	make test_notebooks

.PHONY: test
test:
	$(call activate_conda_env,); \
		pytest --cov -rfsxEX --cov-report term-missing

.PHONY: test_notebooks
test_notebooks:
	$(call activate_conda_env,); \
		pytest -rfsxEX --nbval $(NOTEBOOKS_DIR) --sanitize $(NOTEBOOKS_SANITIZE_FILE)

.PHONY: flake8
flake8:
	$(call activate_conda_env,); \
		flake8 src tests

.PHONY: black
black:
	@status=$$(git status --porcelain pymagicc tests); \
	if test "x$${status}" = x; then \
		$(call activate_conda_env,); \
		black --exclude _version.py --py36 setup.py src tests docs/source/conf.py; \
	else \
		echo Not trying any formatting. Working directory is dirty ... >&2; \
	fi;

.PHONY: new_release
new_release:
	@echo 'For a new release on PyPI:'
	@echo 'git tag vX.Y.Z'
	@echo 'make publish-on-pypi'

.PHONY: release-on-conda
release-on-conda:
	@echo 'For now, this is all very manual'
	@echo 'Checklist:'
	@echo '- version number'
	@echo '- sha'
	@echo '- README.rst badge'
	@echo '- CHANGELOG.rst up to date'

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
	# file solution
	$(call activate_conda,); \
		conda config --add channels conda-forge; \
		conda create -y -n $(CONDA_ENV_NAME); \
		conda activate $(CONDA_ENV_NAME); \
		conda install -y --file $(CONDA_ENV_MINIMAL_YML); \
		conda install -y --file $(CONDA_ENV_DEV_YML); \
		pip install --upgrade pip; \
		pip install -Ur $(PIP_REQUIREMENTS_MINIMAL); \
		pip install -Ur $(PIP_REQUIREMENTS_DEV); \
		pip install -e .

.PHONY: clean_docs
clean_docs:
	rm -rf $(DOCS_DIR)/build

.PHONY: clean
clean:
	$(call activate_conda,); \
		conda deactivate; \
		conda remove --name $(CONDA_ENV_NAME) --all -y
	make clean_docs

.PHONY: variables
variables:
	@echo PWD: $(PWD)
	@echo OS: $(OS)
	@echo SHELL: $(SHELL)

	@echo CONDA_ENV_NAME: $(CONDA_ENV_NAME)
	@echo CONDA_ENV_PATH: $(CONDA_ENV_PATH)
	@echo CONDA_ENV_MINIMAL_YML: $(CONDA_ENV_MINIMAL_YML)
	@echo CONDA_ENV_DEV_YML: $(CONDA_ENV_DEV_YML)

	@echo PIP_REQUIREMENTS_MINIMAL: $(PIP_REQUIREMENTS_MINIMAL)
	@echo PIP_REQUIREMENTS_DEV: $(PIP_REQUIREMENTS_DEV)

	@echo NOTEBOOKS_DIR: $(NOTEBOOKS_DIR)
	@echo NOTEBOOKS_SANITIZE_FILE: $(NOTEBOOKS_SANITIZE_FILE)

	@echo DOCS_DIR: $(DOCS_DIR)
	@echo LATEX_LOGO: $(LATEX_LOGO)
