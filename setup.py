import versioneer

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


PACKAGE_NAME = "netcdf-scm"
AUTHOR = "Zebedee Nicholls"
EMAIL = "zebedee.nicholls@climate-energy-college.org"
URL = "https://github.com/znicholls/netcdf-scm"

DESCRIPTION = (
    "Python wrapper for processing netCDF files for use with simple climate models"
)
README = "README.rst"

SOURCE_DIR = "src"

REQUIREMENTS = [
    "numpy",
    "pandas",
    "python-dateutil",
    "progressbar2",
    "expectexception",
    "openscm>=0.1.0a",
]
REQUIREMENTS_TESTS = ["codecov", "pytest-cov", "pytest>=4.0", "notebook", "nbval"]
REQUIREMENTS_DOCS = ["sphinx>=1.4", "sphinx_rtd_theme"]
REQUIREMENTS_DEPLOY = ["twine>=1.11.0", "setuptools>=38.6.0", "wheel>=0.31.0"]

requirements_dev = [
    *["flake8", "black", "pyam-iamc>=0.2.0"],
    *REQUIREMENTS_TESTS,
    *REQUIREMENTS_DOCS,
    *REQUIREMENTS_DEPLOY,
]

requirements_extras = {
    "docs": REQUIREMENTS_DOCS,
    "tests": REQUIREMENTS_TESTS,
    "deploy": REQUIREMENTS_DEPLOY,
    "dev": requirements_dev,
}

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
    license="2-Clause BSD License",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
    keywords=[
        "netcdf",
        "python",
        "climate",
        "atmosphere",
        "simple climate model",
        "reduced complexity climate model",
    ],
    packages=find_packages(SOURCE_DIR),  # no tests/docs in `src` so don't need exclude
    package_dir={"": SOURCE_DIR},
    # include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require=requirements_extras,
    cmdclass=cmdclass,
)
