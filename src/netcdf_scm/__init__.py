"""
NetCDF-SCM, a package for preparing netCDF files for use in Simple Climate Models (SCMs)

The package builds on the `iris <https://github.com/SciTools/iris>`_ package.
"""
import logging

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

logging.getLogger(__name__).addHandler(logging.NullHandler())
