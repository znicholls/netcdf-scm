Changelog
=========

master
------


v0.6.1
------

- (`#40 <https://github.com/znicholls/netcdf-scm/pull/40>`_) Upgrade to pyam v0.2.0
- (`#38 <https://github.com/znicholls/netcdf-scm/pull/38>`_) Update to using openscm releases and hence drop Python3.6 support
- (`#37 <https://github.com/znicholls/netcdf-scm/pull/37>`_) Adjusted read in of gregorian with 0 reference to give all data from year 1 back
- (`#35 <https://github.com/znicholls/netcdf-scm/pull/35>`_) Fixed bug which prevented SCMCube from crunching to scm timeseries with default earth radius when areacella cube was missing
- (`#34 <https://github.com/znicholls/netcdf-scm/pull/34>`_) Move to new openscm naming i.e. returning ScmDataFrame rather than OpenSCMDataFrameBase
- (`#32 <https://github.com/znicholls/netcdf-scm/pull/32>`_) Move to returning OpenSCMDataFrameBase rather than pandas DataFrame when crunching to scm format
- (`#29 <https://github.com/znicholls/netcdf-scm/pull/29>`_) Fixed bug identified in `#30 <https://github.com/znicholls/netcdf-scm/issues/30>`_
- (`#29 <https://github.com/znicholls/netcdf-scm/pull/29>`_) Put crunching script into formal testsuite which confirms results against KNMI data available `here <https://climexp.knmi.nl/cmip5_indices.cgi?id=someone@somewhere>`_, however no docs or formal example until `#6 <https://github.com/znicholls/netcdf-scm/issues/6>`_ is closed
- (`#28 <https://github.com/znicholls/netcdf-scm/pull/28>`_) Added cmip5 crunching script example, not tested so use with caution until `#6 <https://github.com/znicholls/netcdf-scm/issues/6>`_ is closed

v0.5.1
------

- (`#26 <https://github.com/znicholls/netcdf-scm/pull/26>`_) Expose directory and filename parsers directly


v0.4.3
------

- Move ``import cftime`` into same block as iris imports


v0.4.2
------

- Update ``setup.py`` to install dependencies so that non-Iris dependent functionality can be run from a pip install


v0.4.1
------

- (`#23 <https://github.com/znicholls/netcdf-scm/pull/23>`_) Added ability to handle cubes with invalid calendar (e.g. CMIP6 historical concentrations cubes)
- (`#20 <https://github.com/znicholls/netcdf-scm/pull/20>`_) Added ``CMIP6Input4MIPsCube`` and ``CMIP6OutputCube`` which add compatibility with CMIP6 data


v0.3.1
------

- (`#17 <https://github.com/znicholls/netcdf-scm/pull/17>`_) Update to crunch global and hemispheric means even if land-surface fraction data is missing
- (`#16 <https://github.com/znicholls/netcdf-scm/pull/16>`_) Tidy up experimental crunching script
- (`#15 <https://github.com/znicholls/netcdf-scm/pull/15>`_) Add ability to load from a directory with data that is saved in multiple timeslice files, also adds:

    - adds regular expressions section to development part of docs
    - adds an example script of how to crunch netCDF files into SCM csvs

- (`#14 <https://github.com/znicholls/netcdf-scm/pull/14>`_) Streamline install process
- (`#13 <https://github.com/znicholls/netcdf-scm/pull/13>`_) Add ``load_from_path`` method to ``SCMCube``
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
