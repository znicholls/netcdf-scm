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
        "Programming Language :: Python :: 3.6",
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
    # install_requires=[]
    extras_require={
        "docs": [
            "sphinx",
            "sphinx_rtd_theme",
            "numpy",
            "pandas",
            "pymagicc==2.0.0-alpha",
            "python-dateutil",
            "progressbar2",
        ],
        "test": ["nbresuse", "nbval", "codecov", "pytest-cov", "pytest"],
        "deploy": ["twine", "setuptools", "wheel", "flake8", "black", "versioneer"],
    },
    cmdclass=cmdclass,
)
