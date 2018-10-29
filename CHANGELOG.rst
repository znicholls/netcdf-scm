Changelog
=========

master
------

- (`#12 <https://github.com/znicholls/netcdf-scm/pull/12>`_) Update to use output format that is compatible with pyam
- Update ``netcdftime`` to ``cftime`` to track name change
- (`#10 <https://github.com/znicholls/netcdf-scm/pull/10>`_) Add land/ocean and hemisphere splits to ``_get_scm_masks`` outputs


v0.2.4
------

- Include simple tests in package


v0.2.3
------

- Include LICENSE in package


v0.2.2
------

- Add conda dev environment details


v0.2.1
------

- Update setup.py to reflect actual supported python versions


v0.2.0
------

- (`#4 <https://github.com/znicholls/netcdf-scm/pull/4>`_) Add work done elsewhere previously
    - ``SCMCube`` base class for handling netCDF files
        - reading, cutting and manipulating files for SCM use
    - ``MarbleCMIP5Cube`` for handling CMIP5 netCDF files within a particular directory structure
    - automatic loading and use of surface land fraction and cell area files
    - returns timeseries data, once processed, in pandas DataFrames rather than netCDF format for easier use
    - demonstration notebook of how this first step works
    - CI for entire repository including notebooks
    - automatic documentation with Sphinx


v0.0.1
------

- initial release


v0.0
----

- dummy release
