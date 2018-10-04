import versioneer

import os
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


path = os.path.abspath(os.path.dirname(__file__))


class NetCDFSCM(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        pytest.main(self.test_args)


with open(os.path.join(path, "README.md"), "r") as f:
    readme = f.read()


cmdclass = versioneer.get_cmdclass()
cmdclass.update({"test": NetCDFSCM})

setup(
    name="netcdf-scm",
    version=versioneer.get_version(),
    description="Python wrapper for processing netCDF files for use with simple cliamte models",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Zebedee Nicholls",
    author_email="zebedee.nicholls@climate-energy-college.org",
    url="https://github.com/znicholls/netcdf-scm",
    license=" ",
    keywords=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["tests"]),
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
    # include_package_data=True,
    # install_requires=["pandas", "f90nml"],
    tests_require=["pytest"],
    cmdclass=cmdclass,
)
