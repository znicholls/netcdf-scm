Usage
=====

.. contents:: Contents
    :local:

All of our usage examples are included in ``netcdf-scm/notebooks``.
Command line usage examples can be found in the help message of our command line interface (which we're still building, see `this issue <https://github.com/znicholls/netcdf-scm/issues/6>`_).


Data reference syntax
---------------------

The data reference syntax expected by a cube can be hard to remember.
From a cube, it can be easily queried in two ways:

#. Look at the cube's docstring
#. Look at the default output of the cubes ``get_filepath_from_load_data_from_identifiers_args`` method and the output of the same method with the relevant 'time period/range' argument.


Quick check examples
~~~~~~~~~~~~~~~~~~~~

CMIP6OutputCube
+++++++++++++++

.. code:: python

    >>> from netcdf_scm.iris_cube_wrappers import CMIP6OutputCube
    >>> CMIP6OutputCube().get_filepath_from_load_data_from_identifiers_args()
    './CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/s1960-r2i1p1f3/pr/gn/v20160215/pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f3_gn.nc'

    >>> CMIP6OutputCube().get_filepath_from_load_data_from_identifiers_args(time_range="YYYY-YYYY")
    './CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/s1960-r2i1p1f3/pr/gn/v20160215/pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f3_gn_YYYY-YYYY.nc'

    # printing the docstring providss extra information, including the link to the
    # data reference syntax page (between the carets i.e. between the `<` and `>`)
    >>> print(CMIP6OutputCube.__doc__)
    Subclass of ``SCMCube`` which can be used with CMIP6 model output data

        The data must match the CMIP6 data reference syntax as specified in the 'File name
        template' and 'Directory structure template' sections of the `CMIP6 Data Reference
        Syntax <https://goo.gl/v1drZl>`_.


CMIP6Input4MIPsCube
+++++++++++++++++++

.. code:: python

    >>> from netcdf_scm.iris_cube_wrappers import CMIP6Input4MIPsCube
    >>> CMIP6Input4MIPsCube().get_filepath_from_load_data_from_identifiers_args()
    './input4MIPs/CMIP6/ScenarioMIP/UoM/UoM-REMIND-MAGPIE-ssp585-1-2-0/atmos/yr/mole-fraction-of-carbon-dioxide-in-air/gr1-GMNHSH/1-2-0/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-REMIND-MAGPIE-ssp585-1-2-0_gr1-GMNHSH.nc'

    >>> CMIP6Input4MIPsCube().get_filepath_from_load_data_from_identifiers_args(time_range="YYYY-YYYY")
    './input4MIPs/CMIP6/ScenarioMIP/UoM/UoM-REMIND-MAGPIE-ssp585-1-2-0/atmos/yr/mole-fraction-of-carbon-dioxide-in-air/gr1-GMNHSH/1-2-0/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-REMIND-MAGPIE-ssp585-1-2-0_gr1-GMNHSH_YYYY-YYYY.nc'

    >>> print(CMIP6Input4MIPsCube.__doc__)
    Subclass of ``SCMCube`` which can be used with CMIP6 input4MIPs data

        The data must match the CMIP6 Forcing Datasets Summary, specifically the `Forcing Dataset Specifications <http://goo.gl/r8up31>`_.


MarbleCMIP5Cube
+++++++++++++++

.. code:: python

    >>> from netcdf_scm.iris_cube_wrappers import MarbleCMIP5Cube
    >>> MarbleCMIP5Cube().get_filepath_from_load_data_from_identifiers_args()
    './cmip5/1pctCO2/Amon/tas/CanESM2/r1i1p1/tas_Amon_CanESM2_1pctCO2_r1i1p1.nc'

    >>> MarbleCMIP5Cube().get_filepath_from_load_data_from_identifiers_args(time_period="YYYY-YYYY")
    './cmip5/1pctCO2/Amon/tas/CanESM2/r1i1p1/tas_Amon_CanESM2_1pctCO2_r1i1p1_YYYY-YYYY.nc'

    >>> print(MarbleCMIP5Cube.__doc__)
    Subclass of ``SCMCube`` which can be used with the ``cmip5`` directory on marble.

        This directory structure is very similar, but not quite identical, to the
        recommended CMIP5 directory structure described in section 3.1 of the `CMIP5 Data
        Reference Syntax
        <https://cmip.llnl.gov/cmip5/docs/cmip5_data_reference_syntax_v1-00_clean.pdf>`_.


Masking
-------

In a masked array, the convention is that a value of ``True`` in a mask indicates that the value is invalid.
Hence a ``land_mask`` should be ``False`` where the associated data is land and ``True`` where the associated data is ocean.
This can be confusing as then a land and northern hemisphere mask have to be combined with 'or' logic rather than 'and' logic i.e. ``land_nh_mask = land_mask or nh_mask``, not ``land_nh_mask = land_mask and nh_mask`` as one might intuitively expect.
(One way to think about it is that we want to mask where we're not on land or where we're not in the northern hemisphere, leaving only the regions where we're both on land and in the northern hemisphere).
