Changelog
=========

master
------

- (`#111 <https://github.com/znicholls/netcdf-scm/pull/111>`_) Write tuningstrucs with data in columns rather than rows
- (`#108 <https://github.com/znicholls/netcdf-scm/pull/108>`_) Optimise wranglers and add regression tests
- (`#107 <https://github.com/znicholls/netcdf-scm/pull/107>`_) Add wrangling options for average/point start/mid/end year time manipulations for ``.MAG`` and ``.IN`` files
- (`#106 <https://github.com/znicholls/netcdf-scm/pull/106>`_) Upgrade to new Pymagicc release
- (`#105 <https://github.com/znicholls/netcdf-scm/pull/105>`_) Upgrade to new Pylint release
- (`#104 <https://github.com/znicholls/netcdf-scm/pull/104>`_) Allow wranglers to also handle unit conversions (see `#101 <https://github.com/znicholls/netcdf-scm/pull/101>`_)
- (`#102 <https://github.com/znicholls/netcdf-scm/pull/102>`_) Keep effective area as metadata when calculating SCM timeseries (see `#100 <https://github.com/znicholls/netcdf-scm/pull/100>`_)
- (`#99 <https://github.com/znicholls/netcdf-scm/pull/99>`_) Switch to BSD-3-Clause license
- (`#98 <https://github.com/znicholls/netcdf-scm/pull/98>`_) Add support for reading CMIP6 concentration GMNHSH data
- (`#97 <https://github.com/znicholls/netcdf-scm/pull/97>`_) Add support for tuningstruc data which has been transposed
- (`#95 <https://github.com/znicholls/netcdf-scm/pull/95>`_) Add support for CO2 flux data (fgco2) reading, in the process simplifying crunching and improving lazy weights
- (`#92 <https://github.com/znicholls/netcdf-scm/pull/92>`_) Shrink test files (having moved entire repository to use git lfs properly)
- (`#90 <https://github.com/znicholls/netcdf-scm/pull/90>`_) Rely on iris for lazy crunching
- (`#89 <https://github.com/znicholls/netcdf-scm/pull/89>`_) Change crunching thresholds to be based on data size rather than number of years
- (`#88 <https://github.com/znicholls/netcdf-scm/pull/88>`_) Fix bug when reading more than one multi-dimensional file in a directory
- (`#87 <https://github.com/znicholls/netcdf-scm/pull/87>`_) Add support for crunching data with a height co-ordinate
- (`#84 <https://github.com/znicholls/netcdf-scm/pull/84>`_) Add ability to crunch land, ocean and atmosphere data separately (and sensibly)
- (`#82 <https://github.com/znicholls/netcdf-scm/pull/82>`_) Prepare to add land data handling
- (`#81 <https://github.com/znicholls/netcdf-scm/pull/81>`_) Refactor masks to use weighting instead of masking, doing all the renaming in the process
- (`#80 <https://github.com/znicholls/netcdf-scm/pull/80>`_) Refactor to avoid ``import conftest`` in tests
- (`#77 <https://github.com/znicholls/netcdf-scm/pull/77>`_) Refactor ``netcdf_scm.masks.get_area_mask`` logic to make multi-dimensional co-ordinate support easier
- (`#75 <https://github.com/znicholls/netcdf-scm/pull/75>`_) Check ``land_mask_threshold`` is sensible when retrieving land mask (automatically update if not)
- (`#72 <https://github.com/znicholls/netcdf-scm/pull/72>`_) Monkey patch iris to speed up crunching and go back to linear regridding of default sftlf mask
- (`#74 <https://github.com/znicholls/netcdf-scm/pull/74>`_) Fix bug in mask generation
- (`#70 <https://github.com/znicholls/netcdf-scm/pull/70>`_) Dynamically decide whether to handle data lazily (fix regression tests in process)
- (`#69 <https://github.com/znicholls/netcdf-scm/pull/69>`_) Add El Nino 3.4 mask
- (`#67 <https://github.com/znicholls/netcdf-scm/pull/67>`_) Fix crunching filenaming, tidy up more and add catch for IPSL ``time_origin`` time variable attribute
- (`#66 <https://github.com/znicholls/netcdf-scm/pull/66>`_) Add devops tools and refactor to pass new standards
- (`#64 <https://github.com/znicholls/netcdf-scm/pull/64>`_) Update logging to make post analysis easier and output clearer
- (`#63 <https://github.com/znicholls/netcdf-scm/pull/63>`_) Switch to using cmor name for variable in SCM timeseries output and put standard name in standard_variable_name
- (`#62 <https://github.com/znicholls/netcdf-scm/pull/62>`_) Add netcdf-scm format and crunch to this by default
- (`#61 <https://github.com/znicholls/netcdf-scm/pull/61>`_) Add land fraction when crunching scm timeseries cubes
- (`#58 <https://github.com/znicholls/netcdf-scm/pull/58>`_) Lock tuningstruc wrangling so it can only wrangle to flat tuningstrucs, also includes:

    - turning off all wrangling in preparation for re-doing crunching format
    - adding default sftlf cube

- (`#55 <https://github.com/znicholls/netcdf-scm/pull/55>`_) Hotfix docs so they build properly
- (`#50 <https://github.com/znicholls/netcdf-scm/pull/50>`_) Make pyam-iamc a core dependency

v1.0.0
------

- (`#49 <https://github.com/znicholls/netcdf-scm/pull/49>`_) Make bandit only check ``src``
- (`#48 <https://github.com/znicholls/netcdf-scm/pull/48>`_) Add ``isort`` to checks
- (`#47 <https://github.com/znicholls/netcdf-scm/pull/47>`_) Add regression tests on crunching output to ensure stability. Also:

    - fixes minor docs bug
    - updates default regexp option in crunch and wrangle to avoid ``fx`` files
    - refactors ``cli.py`` a touch to reduce duplication
    - avoids ``collections`` deprecation warning in ``mat4py``


- (`#46 <https://github.com/znicholls/netcdf-scm/pull/46>`_) Fix a number of bugs in ``netcdf-scm-wrangle``'s data handling when converting to tuningstrucs
- (`#45 <https://github.com/znicholls/netcdf-scm/pull/45>`_) Refactor the masking of regions into a module allowing for more regions to be added as needed

v0.7.3
------

- (`#44 <https://github.com/znicholls/netcdf-scm/pull/44>`_) Speed up crunching by forcing data to load before applying masks, not each time a mask is applied

v0.7.2
------

- (`#43 <https://github.com/znicholls/netcdf-scm/pull/43>`_) Speed up crunching, in particular remove string parsing to convert cftime to python datetime

v0.7.1
------

- (`#42 <https://github.com/znicholls/netcdf-scm/pull/42>`_) Add ``netcdf-scm-wrangle`` command line interface
- (`#41 <https://github.com/znicholls/netcdf-scm/pull/41>`_) Fixed bug in path handling of ``CMIP6OutputCube``

v0.6.2
------

- (`#39 <https://github.com/znicholls/netcdf-scm/pull/39>`_) Add ``netcdf-scm-crunch`` command line interface

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
