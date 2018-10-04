import versioneer

import os
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


PACKAGE_NAME = "netcdf-scm"
AUTHOR = "Zebedee Nicholls"
EMAIL= "zebedee.nicholls@climate-energy-college.org"
URL = "https://github.com/znicholls/netcdf-scm"

DESCRIPTION = "Python wrapper for processing netCDF files for use with simple cliamte models"
README = "README.md"

SOURCE_DIR = 'src'

with open(README, 'r') as readme_file:
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
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=" ",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords=[
        'netcdf', 'python', 'climate', 'atmosphere', 'simple climate model',
        'reduced complexity climate model',
    ],
    packages=find_packages(SOURCE_DIR, exclude=["tests"]),
    package_dir={'': SOURCE_DIR},
    # package_data={
    #     "": ["*.csv"],
    #     "pymagicc": [
    #         "MAGICC6/*.txt",
    #         "MAGICC6/out/.gitkeep",
    #         "MAGICC6/run/*.CFG",
    #         "MAGICC6/run/*.exe",
    #         "MAGICC6/run/*.IN",
    #         "MAGICC6/run/*.MON",
    #         "MAGICC6/run/*.prn",
    #         "MAGICC6/run/*.SCEN",
    #     ],
    # },
    include_package_data=True,
    # install_requires=["pandas", "f90nml"],
    tests_require=["pytest"],
    cmdclass=cmdclass,
)
