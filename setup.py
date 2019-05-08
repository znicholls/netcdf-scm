import versioneer

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


PACKAGE_NAME = "netcdf-scm"
DESCRIPTION = "Processing netCDF files for use with simple climate models"
KEYWORDS = [
    "netcdf",
    "netCDF",
    "python",
    "climate",
    "atmosphere",
    "simple climate model",
    "reduced complexity climate model",
    "data processing",
]

AUTHOR = "Zebedee Nicholls"
EMAIL = "zebedee.nicholls@climate-energy-college.org"
URL = "https://github.com/znicholls/netcdf-scm"
PROJECT_URLS = {
    "Bug Reports": "https://github.com/znicholls/netcdf-scm/issues",
    "Documentation": "https://openscm.readthedocs.io/en/latest",
    "Source": "https://github.com/znicholls/netcdf-scm",
}
LICENSE = "2-Clause BSD License"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: BSD License",
    "Intended Audience :: Developers",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.7",
]

ENTRY_POINTS = {"console_scripts": ["netcdf-scm-crunch = netcdf_scm.cli:crunch_data"]}


REQUIREMENTS_INSTALL = [
    "numpy",
    "pandas",
    "python-dateutil",
    "progressbar2",
    "expectexception",
    "openscm>=0.1.0a",
    "click",
]
REQUIREMENTS_TESTS = ["codecov", "pytest-cov", "pytest>=4.0"]
REQUIREMENTS_NOTEBOOKS = ["pyam-iamc>=0.2.0", "notebook", "nbval"]
REQUIREMENTS_DOCS = ["sphinx>=1.4", "sphinx_rtd_theme", "sphinx-click"]
REQUIREMENTS_DEPLOY = ["twine>=1.11.0", "setuptools>=38.6.0", "wheel>=0.31.0"]
requirements_dev = [
    *["flake8", "black"],
    *REQUIREMENTS_TESTS,
    *REQUIREMENTS_NOTEBOOKS,
    *REQUIREMENTS_DOCS,
    *REQUIREMENTS_DEPLOY,
]

REQUIREMENTS_EXTRAS = {
    "docs": REQUIREMENTS_DOCS,
    "tests": REQUIREMENTS_TESTS,
    "notebooks": REQUIREMENTS_NOTEBOOKS,
    "deploy": REQUIREMENTS_DEPLOY,
    "dev": requirements_dev,
}

SOURCE_DIR = "src"

README = "README.rst"

with open(README, "r") as readme_file:
    README_TEXT = readme_file.read()


class NetCDFSCM(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        pytest.main(self.test_args)


cmdclass = versioneer.get_cmdclass()
cmdclass.update({"test": NetCDFSCM})

setup(
    name=PACKAGE_NAME,
    version=versioneer.get_version(),
    description=DESCRIPTION,
    long_description=README_TEXT,
    long_description_content_type="text/x-rst",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=find_packages(SOURCE_DIR),  # no tests/docs in `src` so don't need exclude
    package_dir={"": SOURCE_DIR},
    # include_package_data=True,
    install_requires=REQUIREMENTS_INSTALL,
    extras_require=REQUIREMENTS_EXTRAS,
    cmdclass=cmdclass,
    entry_points=ENTRY_POINTS,
)
