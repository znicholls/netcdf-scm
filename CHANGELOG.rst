Changelog
=========

master
------

- Include simple tests in package


0.2.3
-----

- Include LICENSE in package


0.2.2
-----

- Add conda dev environment details


0.2.1
-----

- Update setup.py to reflect actual supported python versions


0.2.0
-----

- (`#4 <https://github.com/znicholls/netcdf-scm/pull/4>`_) Add work done elsewhere previously
    - ``SCMCube`` base class for handling netCDF files
        - reading, cutting and manipulating files for SCM use
    - ``MarbleCMIP5Cube`` for handling CMIP5 netCDF files within a particular directory structure
    - automatic loading and use of surface land fraction and cell area files
    - returns timeseries data, once processed, in pandas DataFrames rather than netCDF format for easier use
    - demonstration notebook of how this first step works
    - CI for entire repository including notebooks
    - automatic documentation with Sphinx


0.0.1
-----

- initial release


0.0
---

- dummy release
