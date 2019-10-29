"""
Wrappers of the iris cube.

These classes automate handling of a number of netCDF processing steps.
For example, finding surface land fraction files, applying regions to data and
returning timeseries in key regions for simple climate models.
"""
import logging
import os
import re
import warnings
from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime
from os.path import basename, dirname, join, splitext

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scmdata import ScmDataFrame

from . import __version__
from .definitions import _LAND_FRACTION_REGIONS, _SCM_TIMESERIES_META_COLUMNS
from .utils import (
    _check_cube_and_adjust_if_needed,
    _vector_cftime_conversion,
    assert_all_time_axes_same,
    cube_lat_lon_grid_compatible_with_array,
    get_cube_timeseries_data,
    get_scm_cube_time_axis_in_calendar,
    take_lat_lon_mean,
    unify_lat_lon,
)
from .weights import DEFAULT_REGIONS, CubeWeightCalculator

try:
    import cftime

    import iris
    import iris.analysis.cartography
    import iris.coord_categorisation
    import iris.experimental.equalise_cubes
    from iris.exceptions import CoordinateMultiDimError, ConcatenateError
    from iris.fileformats import netcdf
    from iris.util import unify_time_units

    # monkey patch netCDF4 loading to avoid very small chunks
    # required until there is a resolution to
    # https://github.com/SciTools/iris/issues/3333
    # and
    # https://github.com/SciTools/iris/issues/3357
    def _get_cf_var_data(cf_var, filename):
        import netCDF4  # pylint:disable=import-outside-toplevel

        # Get lazy chunked data out of a cf variable.
        dtype = netcdf._get_actual_dtype(cf_var)  # pylint:disable=protected-access

        # Create cube with deferred data, but no metadata
        fill_value = getattr(
            cf_var.cf_data,
            "_FillValue",
            netCDF4.default_fillvals[cf_var.dtype.str[1:]],  # pylint:disable=no-member
        )
        proxy = netcdf.NetCDFDataProxy(
            cf_var.shape, dtype, filename, cf_var.cf_name, fill_value
        )
        return netcdf.as_lazy_data(proxy, chunks=None)

    netcdf._get_cf_var_data = _get_cf_var_data  # pylint:disable=protected-access

except ModuleNotFoundError:  # pragma: no cover # emergency valve
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()

logger = logging.getLogger(__name__)


class SCMCube:  # pylint:disable=too-many-public-methods
    """
    Class for processing netCDF files for use in simple climate models.

    Common, shared operations are implemented here.
    However, methods like ``_get_data_directory`` raise ``NotImplementedError``
    because these are always context dependent.
    Hence to use this base class, you must use a subclass of it which defines these
    context specific methods.
    """

    cube = None
    """
    :obj:`iris.cube.Cube`: The Iris cube which is wrapped by this :obj:`SCMCube` instance.
    """

    lat_name = "latitude"
    """str: The expected name of the latitude co-ordinate in data."""

    lon_name = "longitude"
    """str: The expected name of the longitude co-ordinate in data."""

    time_name = "time"
    """str: The expected name of the time co-ordinate in data."""

    time_period_separator = "-"
    """
    str: Character used to separate time period strings in the time period indicator in filenames.

    e.g. ``-`` is the 'time period separator' in "2015-2030".
    """

    _time_period_regex = None

    _known_timestamps = {
        4: {"datetime_str": "%Y", "expected_timestep": relativedelta(years=+1)},
        6: {"datetime_str": "%Y%m", "expected_timestep": relativedelta(months=+1)},
        8: {"datetime_str": "%Y%m%d", "expected_timestep": relativedelta(days=+1)},
        10: {"datetime_str": "%Y%m%d%H", "expected_timestep": relativedelta(hours=+1)},
    }
    _timestamp_definitions = None

    _scm_timeseries_id_map = {
        "climate_model": "model",
        "scenario": "experiment",
        "activity_id": "activity",
        "member_id": "ensemble_member",
        "mip_era": "mip_era",
        "variable": "variable_name",
    }
    """Mapping from cube attributes (derived from read files) to SCM timeseries metadata"""

    _weight_calculator = None
    """:obj:`CubeWeightCalculator` to use to mask self"""

    _realm_key = "realm"
    """str: key which identifies the realm of the data"""

    _have_guessed_realm = False
    """bool: Have we already guessed our cube data's realm?"""

    _area_weights_units = "m**2"
    """str: assumed units for area weights (checked when data is read)"""

    def __init__(self):
        self._loaded_paths = []
        self._metadata_cubes = {}
        self._weight_calculator = None

    @property
    def netcdf_scm_realm(self):
        """
        str: The realm in which NetCDF-SCM thinks the data belongs.

        This is used to make decisions about how to take averages of the data and
        where to find metadata variables.

        If it is not sure, NetCDF-SCM will guess that the data belongs to the
        'atmosphere' realm.
        """
        try:
            if self.cube.attributes[self._realm_key] in ("ocean", "ocnBgchem"):
                return "ocean"
            if self.cube.attributes[self._realm_key] in ("land"):
                return "land"
            if self.cube.attributes[self._realm_key] in ("atmos"):
                return "atmosphere"
            if not self._have_guessed_realm:
                logger.info(
                    "Unrecognised `%s` attribute value, `%s`, in `self.cube`, NetCDF-SCM will treat the data "
                    "as `atmosphere`",
                    self._realm_key,
                    self.cube.attributes[self._realm_key],
                )
                self._have_guessed_realm = True
            return "atmosphere"
        except KeyError:
            if not self._have_guessed_realm:
                logger.info(
                    "No `%s` attribute in `self.cube`, NetCDF-SCM will treat the data "
                    "as `atmosphere`",
                    self._realm_key,
                )
                self._have_guessed_realm = True
            return "atmosphere"

    @property
    def areacell_var(self):
        """
        str: The name of the variable associated with the area of each gridbox.

        If required, this is used to determine the area of each cell in a data file. For
        example, if our data file is ``tas_Amon_HadCM3_rcp45_r1i1p1_200601.nc`` then
        ``areacell_var`` can be used to work  out the name of the associated cell area
        file. In some cases, it might be as simple as replacing ``tas`` with the value of
        ``areacell_var``.
        """
        if self.netcdf_scm_realm in ("ocean",):
            return "areacello"

        return "areacella"

    @property
    def surface_fraction_var(self):
        """
        str: The name of the variable associated with the surface fraction in each gridbox.

        If required, this is used when looking for the surface fraction file which
        belongs to a given data file. For example, if our data file is
        ``tas_Amon_HadCM3_rcp45_r1i1p1_200601.nc`` then ``surface_fraction_var`` can
        be used to work out the name of the associated surface fraction file. In some
        cases, it might be as simple as replacing ``tas`` with the value of
        ``surface_fraction_var``.
        """
        if self.netcdf_scm_realm in ("ocean",):
            return "sftof"

        return "sftlf"

    @property
    def table_name_for_metadata_vars(self):
        """
        str: The name of the 'table' in which metadata variables can be found.

        For example, ``fx`` or ``Ofx``.

        We wrap this as a property as table typically means ``table_id`` but is
        sometimes referred to in other ways e.g. as ``mip_table`` in CMIP5.
        """
        return self._table_name_for_metadata_vars

    @property
    def _table_name_for_metadata_vars(self):
        if self.netcdf_scm_realm in ("ocean",):
            return "Ofx"

        return "fx"

    @property
    def time_period_regex(self):
        """
        :obj:`_sre.SRE_Pattern`: Regular expression which captures the timeseries identifier in input data files.

        For help on regular expressions, see :ref:`regular expressions <regular-expressions>`.
        """
        if self._time_period_regex is None:
            self._time_period_regex = re.compile(
                r".*_((\d*)" + re.escape(self.time_period_separator) + r"?(\d*)?).*"
            )
        return self._time_period_regex

    @property
    def timestamp_definitions(self):
        """
        dict: Definition of valid timestamp information and corresponding key values.

        This follows the CMIP standards where time strings must be one of the
        following: YYYY, YYYYMM, YYYYMMDD, YYYYMMDDHH or one of the previous combined
        with a hyphen e.g. YYYY-YYYY.

        Each key in the definitions dictionary is the length of the timestamp. Each
        value is itself a dictionary, with keys:

        - datetime_str: the string required to convert a timestamp of this length into a datetime using ``datetime.datetime.strptime``
        - generic_regexp: a regular expression which will match timestamps in this format
        - expected_timestep: a ``dateutil.relativedelta.relativedelta`` object which contains the expected timestep in files with this timestamp

        Examples
        --------
        >>> self.timestamp_definitions[len("2012")]["datetime_str"]
        "%Y"
        """
        if self._timestamp_definitions is None:
            self._timestamp_definitions = {}
            for key, value in self._known_timestamps.items():
                self._timestamp_definitions[key] = value
                generic_regexp = r"\d{" + "{}".format(key) + r"}"
                self._timestamp_definitions[key]["generic_regexp"] = generic_regexp
                hyphen_key = 2 * key + 1
                self._timestamp_definitions[hyphen_key] = {
                    "datetime_str": "{0}-{0}".format(value["datetime_str"]),
                    "expected_timestep": value["expected_timestep"],
                    "generic_regexp": "{0}-{0}".format(generic_regexp),
                }

        return self._timestamp_definitions

    @property
    def dim_names(self):
        """
        list: Names of the dimensions in this cube

        Here the names are the ``standard_names`` which means there can be
        ``None`` in the output.
        """
        return [c.standard_name for c in self.cube.coords()]

    @property
    def lon_dim(self):
        """:obj:`iris.coords.DimCoord` The longitude dimension of the data."""
        return self.cube.coord(self.lon_name)

    @property
    def lon_dim_number(self):
        """
        int: The index which corresponds to the longitude dimension.

        e.g. if longitude is the third dimension of the data, then
        ``self.lon_dim_number`` will be ``2`` (Python is zero-indexed).
        """
        return self.dim_names.index(self.lon_name)

    @property
    def lat_dim(self):
        """:obj:`iris.coords.DimCoord` The latitude dimension of the data."""
        return self.cube.coord(self.lat_name)

    @property
    def lat_dim_number(self):
        """
        int: The index which corresponds to the latitude dimension.

        e.g. if latitude is the first dimension of the data, then
        ``self.lat_dim_number`` will be ``0`` (Python is zero-indexed).
        """
        return self.dim_names.index(self.lat_name)

    @property
    def lat_lon_shape(self):
        """
        tuple: 2D Tuple of ``int`` which gives shape of a lat-lon slice of the data

        e.g. if the cube's shape is (4, 3, 5, 4) and its dimensions are (time, lat,
        depth, lon) then ``cube.lat_lon_shape`` will be ``(3, 4)``
        """
        non_lat_lon_dims = [
            c.standard_name if c.standard_name is not None else c.long_name
            for c in self.cube.coords()
            if c.standard_name not in [self.lat_name, self.lon_name]
        ]

        lat_lon_slice = next(self.cube.slices_over(non_lat_lon_dims))
        return lat_lon_slice.shape

    @property
    def time_dim(self):
        """:obj:`iris.coords.DimCoord` The time dimension of the data."""
        return self.cube.coord(self.time_name)

    @property
    def time_dim_number(self):
        """
        int: The index which corresponds to the time dimension.

        e.g. if time is the first dimension of the data, then
        ``self.time_dim_number`` will be ``0`` (Python is zero-indexed).
        """
        return self.dim_names.index(self.time_name)

    @property
    def info(self):
        """
        dict: Information about the cubes source files

        ``res["files"]`` contains the files used to load the data in this cube.
        ``res["metadata"]`` contains information for each of the metadata cubes used to
        load the data in this cube.
        """
        r = {"files": self._loaded_paths}
        if self._metadata_cubes:
            #  Get the info dict for each of the metadata cubes
            r["metadata"] = {k: v.info for k, v in self._metadata_cubes.items()}
        return r

    def _load_cube(self, filepath, constraint=None):
        logger.debug("loading cube %s", filepath)
        self._loaded_paths.append(filepath)
        # Raises Warning and Exceptions
        self.cube = _check_cube_and_adjust_if_needed(
            iris.load_cube(filepath, constraint=constraint)
        )

    def load_data_from_path(
        self, filepath, process_warnings=True  # pylint:disable=unused-argument
    ):
        """
        Load data from a path.

        If you are using the ``SCMCube`` class directly, this method simply loads the
        path into an iris cube which can be accessed through ``self.cube``.

        If implemented on a subclass of ``SCMCube``, this method should:

        - use ``self.get_load_data_from_identifiers_args_from_filepath`` to determine the suitable set of arguments to pass to ``self.load_data_from_identifiers`` from the filepath
        - load the data using ``self.load_data_from_identifiers`` as this method contains much better checks and helper components

        Parameters
        ----------
        filepath : str
            The filepath from which to load the data.

        process_warnings : bool
            Should I process warnings to add e.g. missing metadata information?
        """
        self._load_cube(filepath)

    def load_data_in_directory(self, directory=None, process_warnings=True):
        """
        Load data in a directory.

        The data is loaded into an iris cube which can be accessed through
        ``self.cube``.

        Initially, this method is intended to only be used to load data when it is
        saved in a number of different timeslice files e.g.:

        - tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc
        - tas_Amon_HadCM3_rcp45_r1i1p1_203101-203512.nc
        - tas_Amon_HadCM3_rcp45_r1i1p1_203601-203812.nc

        It is not intended to be used to load multiple different variables or
        non-continuous timeseries. These use cases could be added in future, but are
        not required yet so have not been included.

        Note that this function removes any attributes which aren't common between the
        loaded cubes. In general, we have found that this mainly means
        ``creation_date``, ``tracking_id`` and ``history`` are deleted. If unsure,
        please check.

        Parameters
        ----------
        directory : str
            Directory from which to load the data.

        process_warnings : bool
            Should I process warnings to add e.g. missing metadata information?

        Raises
        ------
        ValueError
            If the files in the directory are not from the same run (i.e. their filenames are not identical except for the timestamp) or if the files don't form a continuous timeseries.
        """
        self._load_and_concatenate_files_in_directory(
            directory, process_warnings=process_warnings
        )

    def _load_and_concatenate_files_in_directory(
        self, directory, process_warnings=True
    ):
        self._check_data_names_in_same_directory(directory)

        # we use a loop here to make the most of finding missing data like
        # land-surface fraction and cellarea, something iris can't automatically do
        loaded_cubes_iris = iris.cube.CubeList()
        for f in sorted(os.listdir(directory)):
            self.load_data_from_path(
                join(directory, f), process_warnings=process_warnings
            )
            loaded_cubes_iris.append(self.cube)

        unify_time_units(loaded_cubes_iris)
        unify_lat_lon(loaded_cubes_iris)
        iris.experimental.equalise_cubes.equalise_attributes(loaded_cubes_iris)
        try:
            self.cube = loaded_cubes_iris.concatenate_cube()
        except ConcatenateError:
            for ec in loaded_cubes_iris:
                ec.coord("time").attributes.pop("time_origin", None)
            self.cube = loaded_cubes_iris.concatenate_cube()

    def _check_data_names_in_same_directory(self, directory):
        found_files = sorted(os.listdir(directory))

        assertion_error_msg = (
            "Cannot join files in:\n"
            "{}\n"
            "Files found:\n"
            "- {}".format(directory, "\n- ".join(found_files))
        )

        base_regexp = self._get_timestamp_regex_from_filename(found_files[0])
        expected_timestep = self._get_expected_timestep_from_filename(found_files[0])

        file_timestamp_bits_prev = self._get_timestamp_bits_from_filename(
            found_files[0]
        )
        time_format = self.timestamp_definitions[
            len(file_timestamp_bits_prev["timestart_str"])
        ]["datetime_str"]
        for found_file in found_files[1:]:
            if not re.match(base_regexp, found_file):
                raise AssertionError(assertion_error_msg)

            file_timestamp_bits = self._get_timestamp_bits_from_filename(found_file)
            end_time_prev = datetime.strptime(
                file_timestamp_bits_prev["timeend_str"], time_format
            )
            start_time = datetime.strptime(
                file_timestamp_bits["timestart_str"], time_format
            )

            if relativedelta(start_time, end_time_prev) != expected_timestep:
                raise AssertionError(assertion_error_msg)

            file_timestamp_bits_prev = file_timestamp_bits

    def _get_timestamp_regex_from_filename(self, filename):
        timestamp_bits = self._get_timestamp_bits_from_filename(filename)
        timestamp_str = timestamp_bits["timestamp_str"]
        return filename.replace(
            timestamp_str,
            self.timestamp_definitions[len(timestamp_str)]["generic_regexp"],
        )

    def _get_expected_timestep_from_filename(self, filename):
        timestamp_bits = self._get_timestamp_bits_from_filename(filename)
        timestamp_str = timestamp_bits["timestamp_str"]
        return self.timestamp_definitions[len(timestamp_str)]["expected_timestep"]

    def _get_timestamp_bits_from_filename(self, filename):
        regex_matches = re.match(self.time_period_regex, filename)
        timestamp_str = regex_matches.group(1)
        self._check_time_period_valid(timestamp_str)
        start_time = regex_matches.group(2)
        end_time = regex_matches.group(3)
        timestep = self.timestamp_definitions[len(timestamp_str)]["expected_timestep"]
        return {
            "timestamp_str": timestamp_str,
            "timestart_str": start_time,
            "timeend_str": end_time,
            "expected_timestep": timestep,
        }

    def _process_load_data_from_identifiers_warnings(self, w):
        area_cella_warn = "Missing CF-netCDF measure variable 'areacella'"
        area_cello_warn = "Missing CF-netCDF measure variable 'areacello'"
        for warn in w:
            if any(
                [m in str(warn.message) for m in (area_cella_warn, area_cello_warn)]
            ):
                self._add_areacell_measure(warn, self.areacell_var)
            else:
                logger.warning(warn.message)

    def _add_areacell_measure(self, original_warn, area_variable):
        try:
            area_cube = self.get_metadata_cube(area_variable).cube
            area_measure = iris.coords.CellMeasure(
                area_cube.core_data(),
                standard_name=area_cube.standard_name,
                long_name=area_cube.long_name,
                var_name=area_cube.var_name,
                units=area_cube.units,
                attributes=area_cube.attributes,
                measure="area",
            )
            self.cube.add_cell_measure(
                area_measure, data_dims=[self.lat_dim_number, self.lon_dim_number]
            )
        except Exception as e:  # pylint:disable=broad-except
            error_msg = str(
                original_warn.message
            ) + ". Tried to add {} cube but another exception was raised: {}".format(
                area_variable, str(e)
            )
            logger.debug(error_msg)

    def get_metadata_cube(self, metadata_variable, cube=None):
        """
        Load a metadata cube from self's attributes.

        Parameters
        ----------
        metadata_variable : str
            the name of the metadata variable to get, as it appears in the filename.

        cube : :obj:`SCMCube`
            Optionally, pass in an already loaded metadata cube to link it to currently loaded cube

        Returns
        -------
        type(self)
            instance of self which has been loaded from the file containing the metadata variable of interest.

        Raises
        ------
        TypeError
            ``cube`` is not an :obj:`ScmCube`
        """
        if cube is not None:
            if not isinstance(cube, SCMCube):
                raise TypeError("cube must be an SCMCube instance")

            self._metadata_cubes[metadata_variable] = cube

        return self._metadata_cubes[metadata_variable]

    def get_scm_timeseries(
        self,
        surface_fraction_cube=None,
        areacell_scmcube=None,
        regions=None,
        lazy=False,
    ):
        """
        Get SCM relevant timeseries from ``self``.

        Parameters
        ----------
        surface_fraction_cube : :obj:`SCMCube`, optional
            surface fraction data which is defines surface fraction weights. If
            ``None``, we try to load the land surface fraction automatically.

        areacell_scmcube : :obj:`SCMCube`, optional
            cell area data which is used to take the latitude-longitude mean of the
            cube's data. If ``None``, we try to load this data automatically and if
            that fails we fall back onto ``iris.analysis.cartography.area_weights``.

        regions : list[str]
            List of regions to use. If ``None`` then
            ``netcdf_scm.regions.DEFAULT_REGIONS`` is used.

        lazy : bool
            Should I process the data lazily? This can be slow as data has to be read
            off disk multiple time.

        Returns
        -------
        :obj:`openscm.io.ScmDataFrame`
            An OpenSCM DataFrame instance with the data in the ``data`` attribute and
            metadata in the ``metadata`` attribute.
        """
        regions = regions if regions is not None else DEFAULT_REGIONS
        scm_timeseries_cubes = self.get_scm_timeseries_cubes(
            surface_fraction_cube=surface_fraction_cube,
            areacell_scmcube=areacell_scmcube,
            regions=regions,
            lazy=lazy,
        )

        return self.convert_scm_timeseries_cubes_to_openscmdata(scm_timeseries_cubes)

    def get_scm_timeseries_weights(
        self, surface_fraction_cube=None, areacell_scmcube=None, regions=None
    ):
        """
        Get the scm timeseries weights

        Parameters
        ----------
        surface_fraction_cube : :obj:`SCMCube`, optional
            land surface fraction data which is used to determine whether a given
            gridbox is land or ocean. If ``None``, we try to load the land surface fraction automatically.

        areacell_scmcube : :obj:`SCMCube`, optional
            cell area data which is used to take the latitude-longitude mean of the
            cube's data. If ``None``, we try to load this data automatically and if
            that fails we fall back onto ``iris.analysis.cartography.area_weights``.

        regions : list[str]
            List of regions to use. If ``None`` then
            ``netcdf_scm.regions.DEFAULT_REGIONS`` is used.

        Returns
        -------
        dict
            Dictionary of region name-weights key-value pairs
        """
        if areacell_scmcube is not None:
            # set area cube to appropriate variable
            self.get_metadata_cube(self.areacell_var, cube=areacell_scmcube)

        if surface_fraction_cube is not None:
            # set area cube to appropriate variable
            self.get_metadata_cube(
                self.surface_fraction_var, cube=surface_fraction_cube
            )

        if self._weight_calculator is None:
            self._weight_calculator = CubeWeightCalculator(self)

        regions = regions if regions is not None else DEFAULT_REGIONS
        scm_weights = self._weight_calculator.get_weights(regions)

        return scm_weights

    def get_scm_timeseries_cubes(
        self,
        surface_fraction_cube=None,
        areacell_scmcube=None,
        regions=None,
        lazy=False,
    ):
        """
        Get SCM relevant cubes

        The effective areas used for each of the regions are added as auxillary
        co-ordinates of each timeseries cube.

        If global, Northern Hemisphere and Southern Hemisphere land cubes are
        calculated, then three auxillary co-ordinates are also added to each cube:
        ``land_fraction``, ``land_fraction_northern_hemisphere`` and
        ``land_fraction_southern_hemisphere``. These co-ordinates document the area
        fraction that was considered to be land when the cubes were crunched i.e.
        ``land_fraction`` is the fraction of the entire globe which was considered to
        be land, ``land_fraction_northern_hemisphere`` is the fraction of the Northern
        Hemisphere which was considered to be land and
        ``land_fraction_southern_hemisphere`` is the fraction of the Southern
        Hemisphere which was considered to be land.

        Parameters
        ----------
        surface_fraction_cube : :obj:`SCMCube`, optional
            land surface fraction data which is used to determine whether a given
            gridbox is land or ocean. If ``None``, we try to load the land surface fraction automatically.

        areacell_scmcube : :obj:`SCMCube`, optional
            cell area data which is used to take the latitude-longitude mean of the
            cube's data. If ``None``, we try to load this data automatically and if
            that fails we fall back onto ``iris.analysis.cartography.area_weights``.

        regions : list[str]
            List of regions to use. If ``None`` then
            ``netcdf_scm.regions.DEFAULT_REGIONS`` is used.

        lazy : bool
            Should I process the data lazily? This can be slow as data has to be read
            off disk multiple time.

        Returns
        -------
        dict
            Cubes, with latitude-longitude mean data as appropriate for each of the
            SCM relevant regions.
        """
        regions = regions if regions is not None else DEFAULT_REGIONS
        scm_timeseries_weights = self.get_scm_timeseries_weights(
            surface_fraction_cube=surface_fraction_cube,
            areacell_scmcube=areacell_scmcube,
            regions=regions,
        )

        def crunch_timeseries(region, weights):
            scm_cube = take_lat_lon_mean(self, weights)
            scm_cube = self._add_metadata_to_region_timeseries_cube(scm_cube, region)

            ws = weights.shape
            if ws == self.lat_lon_shape or ws[::-1] == self.lat_lon_shape:
                weights_area_slice = weights
            else:
                area_slicer = [
                    slice(None)
                    if i in [self.lat_dim_number, self.lon_dim_number]
                    else 0
                    for i in range(len(weights.shape))
                ]
                weights_area_slice = weights[tuple(area_slicer)]
                wass = weights_area_slice.shape
                confused_shape = (
                    wass != self.lat_lon_shape and wass[::-1] != self.lat_lon_shape
                )
                if confused_shape:  # pragma: no cover
                    raise AssertionError("Can't work out area shapes")

            area = np.sum(weights_area_slice)
            if region == "World" and self.netcdf_scm_realm in ("land", "ocean"):
                # yuck hard-coding to be removed when addressing
                # https://github.com/znicholls/netcdf-scm/issues/103
                area *= 1 / 100  # correct for sftlf weights being 0-100
            elif "Land" in region:
                area *= 1 / 100  # correct for sftlf weights being 0-100
            elif "Ocean" in region or "El Nino N3.4" in region:
                # El Nino N3.4 hard-coding to be removed when addressing
                # https://github.com/znicholls/netcdf-scm/issues/103
                area *= 1 / 100  # correct for sftlf weights being 0-100

            return region, scm_cube, area

        memory_error = False
        if not lazy:
            try:
                crunch_list = self._crunch_in_memory(
                    crunch_timeseries, scm_timeseries_weights
                )
            except MemoryError:
                logger.warning(
                    "Data won't fit in memory, will process lazily (hence slowly)"
                )
                memory_error = True

        if lazy or memory_error:
            if lazy:
                logger.info("Forcing lazy crunching")

            data_dir = dirname(self.info["files"][0])
            self.__init__()
            self.load_data_in_directory(data_dir)
            scm_timeseries_weights = self.get_scm_timeseries_weights(
                surface_fraction_cube=surface_fraction_cube,
                areacell_scmcube=areacell_scmcube,
                regions=regions,
            )
            crunch_list = self._crunch_serial(crunch_timeseries, scm_timeseries_weights)

        timeseries_cubes = {region: ts_cube for region, ts_cube, _ in crunch_list}
        areas = {region: area for region, _, area in crunch_list if area is not None}
        timeseries_cubes = self._add_areas(timeseries_cubes, areas)
        timeseries_cubes = self._add_land_fraction(timeseries_cubes, areas)
        return timeseries_cubes

    def _crunch_in_memory(self, crunch_timeseries, scm_regions):
        # calculating lat-lon mean in parallel could go here
        self._ensure_data_realised()
        logger.debug("Crunching SCM timeseries in memory")
        return self._crunch_serial(crunch_timeseries, scm_regions)

    @staticmethod
    def _crunch_serial(crunch_timeseries, scm_regions):
        return [
            crunch_timeseries(region, weights)
            for region, weights in scm_regions.items()
        ]

    def _ensure_data_realised(self):
        # force the data to realise
        if self.cube.has_lazy_data():
            self.cube.data  # pylint:disable=pointless-statement

    def _add_areas(self, timeseries_cubes, areas):
        for cube in timeseries_cubes.values():
            for region, area in areas.items():
                area_key = self._convert_region_to_area_key(region)
                cube.cube.add_aux_coord(
                    iris.coords.AuxCoord(
                        area, long_name=area_key, units=self._area_weights_units
                    )
                )

        return timeseries_cubes

    @staticmethod
    def _convert_region_to_area_key(region):
        return "area_{}".format(region.lower().replace("|", "_").replace(" ", "_"))

    @staticmethod
    def _add_land_fraction(timeseries_cubes, areas):
        add_land_frac = all([r in areas for r in _LAND_FRACTION_REGIONS])

        if add_land_frac:
            for cube in timeseries_cubes.values():
                extensions = {
                    "land_fraction": "",
                    "land_fraction_northern_hemisphere": "|Northern Hemisphere",
                    "land_fraction_southern_hemisphere": "|Southern Hemisphere",
                }
                fractions = {}
                for k, ext in extensions.items():
                    closed_sum = np.isclose(
                        areas["World{}".format(ext)],
                        areas["World{}|Land".format(ext)]
                        + areas["World{}|Ocean".format(ext)],
                        rtol=1e-3,
                    )
                    if not closed_sum:  # pragma: no cover
                        raise AssertionError(
                            "Ocean and land area sums don't equal total..."
                        )

                    fractions[k] = (
                        areas["World{}|Land".format(ext)] / areas["World{}".format(ext)]
                    )
                fractions = {
                    k: areas["World{}|Land".format(ext)] / areas["World{}".format(ext)]
                    for k, ext in extensions.items()
                }
                for k, v in fractions.items():
                    cube.cube.add_aux_coord(
                        iris.coords.AuxCoord(v, long_name=k, units=1)
                    )
        else:
            logger.warning(
                "Not calculating land fractions as all required cubes are not "
                "available"
            )

        return timeseries_cubes

    def _add_metadata_to_region_timeseries_cube(self, scmcube, region):
        has_root_dir = (
            hasattr(self, "root_dir")  # pylint:disable=no-member
            and self.root_dir is not None  # pylint:disable=no-member
        )
        if has_root_dir:
            source_file_info = "Files: {}".format(
                [
                    p.replace(self.root_dir, "")  # pylint:disable=no-member
                    for p in self.info["files"]
                ]
            )
        else:
            source_file_info = "Files: {}".format(
                [basename(p) for p in self.info["files"]]
            )
        if "metadata" in self.info:
            source_meta = {}
            for k, v in self.info["metadata"].items():
                if has_root_dir:
                    source_meta[k] = [
                        "{}".format(
                            p.replace(self.root_dir, "")  # pylint:disable=no-member
                        )
                        for p in v["files"]
                    ]
                else:
                    source_meta[k] = ["{}".format(basename(p)) for p in v["files"]]
            source_file_info = "; ".join(
                [source_file_info]
                + ["{}: {}".format(k, v) for k, v in source_meta.items()]
            )

        scmcube.cube.attributes[
            "crunch_netcdf_scm_version"
        ] = "{} (more info at github.com/znicholls/netcdf-scm)".format(__version__)
        scmcube.cube.attributes["crunch_source_files"] = source_file_info
        scmcube.cube.attributes["region"] = region
        scmcube.cube.attributes.update(self._get_scm_timeseries_ids())

        return scmcube

    def get_area_weights(self, areacell_scmcube=None):
        """
        Get area weights for this cube

        Parameters
        ----------
        areacell_scmcube : :obj:`ScmCube`
            :obj:`ScmCube` containing areacell data. If ``None``, we calculate the weights using iris.

        Returns
        -------
        np.ndarray
            Weights on the cube's latitude-longitude grid.

        Raises
        ------
        iris.exceptions.CoordinateMultiDimError
            The cube's co-ordinates are multi-dimensional and we don't have cell area
            data.

        ValueError
            Area weights units are not as expected (contradict with
            ``self._area_weights_units``).
        """
        areacell_scmcube = self._get_areacell_scmcube(areacell_scmcube)

        if areacell_scmcube is not None:
            areacell_units = areacell_scmcube.cube.units
            if self._area_weights_units != areacell_units:
                raise ValueError(
                    "Your weights need to be in {} but your areacell cube has units of {}".format(
                        self._area_weights_units, areacell_units
                    )
                )
            area_weights = areacell_scmcube.cube.data
            if cube_lat_lon_grid_compatible_with_array(self, area_weights):
                return area_weights
            logger.exception("Area weights incompatible with lat lon grid")

        logger.warning(
            "Couldn't find/use areacell_cube, falling back to iris.analysis.cartography.area_weights"
        )
        if self._area_weights_units != "m**2":
            raise ValueError(
                "iris.analysis.cartography only returns weights in m**2 but your weights need to be {}".format(
                    self._area_weights_units
                )
            )
        try:
            lat_lon_slice = next(self.cube.slices([self.lat_name, self.lon_name]))
            iris_weights = iris.analysis.cartography.area_weights(lat_lon_slice)
        except ValueError:
            logger.warning("Guessing latitude and longitude bounds")
            try:
                self.cube.coord("latitude").guess_bounds()
                self.cube.coord("longitude").guess_bounds()
            except CoordinateMultiDimError:
                error_msg = (
                    "iris does not yet support multi-dimensional co-ordinates, you "
                    "will need your data's cell area information before you can crunch"
                )
                raise CoordinateMultiDimError(error_msg)
            lat_lon_slice = next(self.cube.slices([self.lat_name, self.lon_name]))
            iris_weights = iris.analysis.cartography.area_weights(lat_lon_slice)

        return iris_weights

    def _get_areacell_scmcube(self, areacell_scmcube):
        try:
            areacell_scmcube = self.get_metadata_cube(
                self.areacell_var, cube=areacell_scmcube
            )
            if not isinstance(areacell_scmcube.cube, iris.cube.Cube):
                logger.warning(
                    "areacell cube which was found has cube attribute which isn't an iris cube"
                )
            else:
                return areacell_scmcube
        except (
            iris.exceptions.ConstraintMismatchError,
            AttributeError,
            TypeError,
            OSError,
            NotImplementedError,
            KeyError,
        ) as e:
            logger.debug("Could not calculate areacell, error message: %s", e)

        return None

    def convert_scm_timeseries_cubes_to_openscmdata(
        self, scm_timeseries_cubes, out_calendar=None
    ):
        """
        Convert dictionary of SCM timeseries cubes to an :obj:`ScmDataFrame`

        Parameters
        ----------
        scm_timeseries_cubes : dict
            Dictionary of "region name"-:obj:`ScmCube` key-value pairs.

        out_calendar : str
            Calendar to use for the time axis of the output :obj:`ScmDataFrame`

        Returns
        -------
        :obj:`ScmDataFrame`
            :obj:`ScmDataFrame` containing the data from the SCM timeseries cubes

        Raises
        ------
        NotImplementedError
            The (original) input data has dimensions other than time, latitude and
            longitude (so the data to convert has dimensions other than time).
        """
        data = []
        columns = {mc: [] for mc in _SCM_TIMESERIES_META_COLUMNS}
        for scm_cube in scm_timeseries_cubes.values():
            try:
                data.append(get_cube_timeseries_data(scm_cube, realise_data=True))
            except AssertionError:
                # blocked until we work out how to handle extra coord information in
                # SCMDataFrame
                raise NotImplementedError(
                    "Cannot yet get SCM timeseries for data with dimensions other "
                    "than time, latitude and longitude"
                )
            for column_name, column_values in columns.items():
                column_values.append(scm_cube.cube.attributes[column_name])

        data = np.vstack(data).T

        time_index, out_calendar = self._get_openscmdata_time_axis_and_calendar(
            scm_timeseries_cubes, out_calendar=out_calendar
        )
        unit = str(self.cube.units).replace("-", "^-")
        if unit == "1":
            unit = "dimensionless"  # ensure units behave with pint

        output = ScmDataFrame(
            data,
            index=time_index,
            columns={**{"unit": str(unit), "model": "unspecified"}, **columns},
        )
        try:
            output.metadata["calendar"] = out_calendar
        except AttributeError:
            output.metadata = {"calendar": out_calendar}

        return self._add_metadata_to_scmdataframe(output, scm_timeseries_cubes)

    @staticmethod
    def _add_metadata_to_scmdataframe(scmdf, scm_timeseries_cubes):
        for i, scm_cube in enumerate(scm_timeseries_cubes.values()):
            for coord in scm_cube.cube.coords():
                if coord.standard_name in ["time", "latitude", "longitude", "height"]:
                    continue

                if coord.long_name.startswith("land_fraction"):
                    new_col = coord.long_name
                    new_val = coord.points.squeeze()
                else:  # pragma: no cover
                    # this is really how it should work for land_fraction too but we don't
                    # have a stable solution for parameter handling in OpenSCMDataFrame yet so
                    # I've done the above instead
                    new_col = "{} ({})".format(coord.long_name, str(coord.units))
                    new_val = float(coord.points.squeeze())

                if not i:
                    scmdf.metadata[new_col] = new_val
                elif scmdf.metadata[new_col] != new_val:  # pragma: no cover
                    raise AssertionError("Cubes have different metadata...")

            for k, v in scm_cube.cube.attributes.items():
                if k in scmdf.meta:
                    continue
                if not i:
                    scmdf.metadata[k] = v
                elif scmdf.metadata[k] != v:  # pragma: no cover
                    raise AssertionError("Cubes have different metadata...")

        return scmdf

    def _get_scm_timeseries_ids(self):
        output = {}
        for k in _SCM_TIMESERIES_META_COLUMNS:
            if k == "region":
                continue  # handled elsewhere
            if k == "variable_standard_name":
                output[k] = self.cube.standard_name
                continue
            attr_to_check = self._scm_timeseries_id_map[k]
            val = getattr(self, attr_to_check) if hasattr(self, attr_to_check) else None
            if k == "variable":
                if val is not None:
                    output[k] = val
                else:
                    warn_msg = (
                        "Could not determine {}, filling with "
                        "standard_name".format(k)
                    )
                    logger.warning(warn_msg)
                    output[k] = self.cube.standard_name
                continue
            if val is not None:
                output[k] = val
            else:
                warn_msg = "Could not determine {}, filling with 'unspecified'".format(
                    k
                )
                logger.warning(warn_msg)
                output[k] = "unspecified"

        return output

    def _get_openscmdata_time_axis_and_calendar(
        self, scm_timeseries_cubes, out_calendar
    ):
        if out_calendar is None:
            out_calendar = self.cube.coords("time")[0].units.calendar

        time_axes = [
            get_scm_cube_time_axis_in_calendar(scm_cube, out_calendar)
            for scm_cube in scm_timeseries_cubes.values()
        ]
        assert_all_time_axes_same(time_axes)
        time_axis = time_axes[0]
        if isinstance(time_axis[0], cftime.datetime):
            # inspired by xarray, should make a PR back in there...
            if out_calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
                logger.warning(
                    "Performing lazy conversion to datetime for calendar: %s. This "
                    "may cause subtle errors in operations that depend on the length "
                    "of time between dates",
                    out_calendar,
                )
            time_axis = _vector_cftime_conversion(time_axis)
        else:
            pass  # leave openscm to handle from here

        # As we sometimes have to deal with long timeseries, we force the index to be
        # pd.Index and not pd.DatetimeIndex. We can't use DatetimeIndex because of a
        # pandas limitation, see
        # http://pandas-docs.github.io/pandas-docs-travis/timeseries.html#timestamp-limitations
        return pd.Index(time_axis, dtype="object", name="time"), out_calendar

    def _check_time_period_valid(self, time_period_str):
        """
        Check that a time_period identifier string is valid.

        Parameters
        ----------
        time_period_str : str
            Time period string to check.

        Raises
        ------
        ValueError
            If the time period is not in a valid format.
        """
        if "_" in time_period_str:
            self._raise_time_period_invalid_error(time_period_str)

        if "-" in time_period_str:
            dates = time_period_str.split("-")
            if (len(dates) != 2) or (len(dates[0]) != len(dates[1])):
                self._raise_time_period_invalid_error(time_period_str)
        else:
            dates = [time_period_str]

        try:
            time_format = self.timestamp_definitions[len(dates[0])]["datetime_str"]
        except KeyError:
            self._raise_time_period_invalid_error(time_period_str)
        for date in dates:
            try:
                datetime.strptime(date, time_format)
            except ValueError:
                self._raise_time_period_invalid_error(time_period_str)

        if len(dates) == 2:
            start_date = datetime.strptime(dates[0], time_format)
            end_date = datetime.strptime(dates[1], time_format)
            if start_date >= end_date:
                self._raise_time_period_invalid_error(time_period_str)

    @staticmethod
    def _raise_time_period_invalid_error(time_period_str):
        message = "Your time_period indicator ({}) does not look right".format(
            time_period_str
        )
        raise ValueError(message)


class _CMIPCube(SCMCube, ABC):
    """Base class for cubes which follow a CMIP data reference syntax"""

    time_period = None
    """
    str: The timespan of the data stored by this cube

    The string follows the cube's data reference syntax.
    """

    filename_bits_separator = "_"
    """
    str: Character used to separate different parts of the filename

    This character is used to separate different metadata in the filename. For
    example, if ``filename_bits_separator`` is an underscore, then the filename could
    have a component like "bcc_1990-2010" and the underscore would separate "bcc" (the
    modelling centre which produced the file) from "1990-2010" (the time period of the
    file).
    """

    def load_data_from_path(self, filepath, process_warnings=True):
        """
        Load data from a path.

        Parameters
        ----------
        filepath : str
            The filepath from which to load the data.

        process_warnings : bool
            Should I process warnings to add e.g. missing metadata information?
        """
        load_data_from_identifiers_args = self.get_load_data_from_identifiers_args_from_filepath(
            filepath
        )
        self.load_data_from_identifiers(
            process_warnings=process_warnings, **load_data_from_identifiers_args
        )

    def _load_and_concatenate_files_in_directory(
        self, directory, process_warnings=True
    ):
        super()._load_and_concatenate_files_in_directory(
            directory, process_warnings=process_warnings
        )
        self._add_time_period_from_files_in_directory(directory)

    def _add_time_period_from_files_in_directory(self, directory):
        self._check_data_names_in_same_directory(directory)

        loaded_files = sorted(os.listdir(directory))
        strt = self._get_timestamp_bits_from_filename(loaded_files[0])["timestart_str"]
        end = self._get_timestamp_bits_from_filename(loaded_files[-1])["timeend_str"]
        self._time_id = self.time_period_separator.join([strt, end])

    def get_load_data_from_identifiers_args_from_filepath(self, filepath):
        """
        Get the set of identifiers to use to load data from a filepath.

        Parameters
        ----------
        filepath : str
            The filepath from which to load the data.

        Returns
        -------
        dict
            Set of arguments which can be passed to
            ``self.load_data_from_identifiers`` to load the data in the filepath.

        Raises
        ------
        ValueError
            Path and filename contradict each other
        """
        path_ids = self.process_path(dirname(filepath))
        name_ids = self.process_filename(basename(filepath))
        for key, value in path_ids.items():
            if (key in name_ids) and (value != name_ids[key]):
                error_msg = (
                    "Path and filename do not agree:\n"
                    "    - path {0}: {1}\n"
                    "    - filename {0}: {2}\n".format(key, value, name_ids[key])
                )
                raise ValueError(error_msg)

        return {**path_ids, **name_ids}

    @abstractmethod
    def process_path(self, path):
        """
        Cut a path into its identifiers

        Parameters
        ----------
        path : str
            The path to process. Path here means just the path, no filename
            should be included.

        Returns
        -------
        dict
            A dictionary where each key is the identifier name and each value is the value of that identifier for the input path
        """

    @abstractmethod
    def process_filename(self, filename):
        """
        Cut a filename into its identifiers

        Parameters
        ----------
        filename : str
            The filename to process. Filename here means just the filename, no path
            should be included.

        Returns
        -------
        dict
            A dictionary where each key is the identifier name and each value is the value of that identifier for the input filename
        """

    def load_data_from_identifiers(self, process_warnings=True, **kwargs):
        """
        Load data using key identifiers.

        The identifiers are used to determine the path of the file to load. The file
        is then loaded into an iris cube which can be accessed through ``self.cube``.

        Parameters
        ----------
        process_warnings : bool
            Should I process warnings to add e.g. missing metadata information?

        kwargs : any
            Arguments which can then be processed by
            ``self.get_filepath_from_load_data_from_identifiers_args`` and
            ``self.get_variable_constraint`` to determine the full
            filepath of the file to load and the variable constraint to use.
        """
        with warnings.catch_warnings(record=True) as w:
            # iris v2.2.0 under py3.7 raises a DeprecationWarning about using collections, see https://github.com/SciTools/iris/pull/3320
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module=r".*collections.*"
            )
            fpath = self.get_filepath_from_load_data_from_identifiers_args(**kwargs)
            self._load_cube(fpath, self.get_variable_constraint())

        if w and process_warnings:
            self._process_load_data_from_identifiers_warnings(w)

    def get_metadata_cube(self, metadata_variable, cube=None):
        """
        Load a metadata cube from self's attributes.

        Parameters
        ----------
        metadata_variable : str
            the name of the metadata variable to get, as it appears in the filename.

        cube : :obj:`SCMCube`
            Optionally, pass in an already loaded metadata cube to link it to currently loaded cube

        Returns
        -------
        type(self)
            instance of self which has been loaded from the file containing the metadata variable of interest.

        Raises
        ------
        TypeError
            ``cube`` is not an :obj:`ScmCube`
        """
        if cube is not None:
            return super().get_metadata_cube(metadata_variable, cube=cube)

        if metadata_variable not in self._metadata_cubes:
            load_args = self._get_metadata_load_arguments(metadata_variable)

            cube = type(self)()
            cube._metadata_cubes = {  # pylint:disable=protected-access
                k: v for k, v in self._metadata_cubes.items() if k != metadata_variable
            }
            cube.load_data_from_identifiers(**load_args)

            return super().get_metadata_cube(metadata_variable, cube=cube)
        return super().get_metadata_cube(metadata_variable)

    @abstractmethod
    def get_filepath_from_load_data_from_identifiers_args(self, **kwargs):
        """
        Get the full filepath of the data to load from the arguments passed to ``self.load_data_from_identifiers``.

        This function should, in most cases, call ``self.get_data_directory`` and
        ``self.get_data_filename``.

        Parameters
        ----------
        **kwargs
            Arguments, initially passed to ``self.load_data_from_identifiers`` from which the full
            filepath of the file to load should be determined.

        Returns
        -------
        str
            The full filepath (path and name) of the file to load.
        """

    def get_variable_constraint(self):
        """
        Get the iris variable constraint to use when loading data with ``self.load_data_from_identifiers``.

        Returns
        -------
        :obj:`iris.Constraint`
            constraint to use which ensures that only the variable of interest is
            loaded.
        """
        # thank you Duncan!!
        # https://github.com/SciTools/iris/issues/2107#issuecomment-246644471
        return iris.Constraint(
            cube_func=(
                lambda c: c.var_name
                == np.str(self._variable_name_for_constraint_loading)
            )
        )

    @classmethod
    def get_data_reference_syntax(cls, **kwargs):
        """
        Get data reference syntax for this cube

        Parameters
        ----------
        kwargs : str
            Attributes of the cube to set before generating the example data reference
            syntax.

        Returns
        -------
        str
            Example of the full path to a file for the given ``kwargs`` with this
            cube's data reference syntax.
        """
        helper = cls()
        for a in dir(helper):
            try:
                if callable(a) or callable(getattr(helper, a)):
                    continue
                if a == "filename_bits_separator" or a.startswith("_"):
                    continue
                new_separator = "-" if cls.filename_bits_separator == "_" else "_"
                setattr(
                    helper, a, a.replace(cls.filename_bits_separator, new_separator)
                )
            except AttributeError:
                continue

        for k, v in kwargs.items():
            setattr(helper, k, v)

        return join(helper.get_data_directory(), helper.get_data_filename())

    def get_data_directory(self):
        """
        Get the path to a data file from self's attributes.

        This can take multiple forms, it may just return a previously set
        filepath attribute or it could combine a number of different metadata
        elements (e.g. model name, experiment name) to create the data path.

        Returns
        -------
        str
            path to the data file from which this cube has been/will be loaded

        Raises
        ------
        OSError
            The data directory cannot be determined
        """
        try:
            return self._get_data_directory()
        except TypeError:  # some required attributes still None
            raise OSError("Could not determine data directory")

    @abstractmethod
    def _get_data_directory(self):
        pass  # pragma: no cover

    def get_data_filename(self):
        """
        Get the name of a data file from self's attributes.

        This can take multiple forms, it may just return a previously set
        filename attribute or it could combine a number of different metadata
        elements (e.g. model name, experiment name) to create the data name.

        Returns
        -------
        str
            name of the data file from which this cube has been/will be loaded.

        Raises
        ------
        OSError
            The data directory cannot be determined
        """
        try:
            return self._get_data_filename()
        except TypeError:  # some required attributes still None
            raise OSError("Could not determine data filename")

    @abstractmethod
    def _get_data_filename(self):
        pass  # pragma: no cover

    @abstractmethod
    def _get_metadata_load_arguments(self, metadata_variable):
        """
        Get the arguments to load a metadata file from self's attributes.

        This can take multiple forms, it may just return a previously set
        metada_filename attribute or it could combine a number of different
        metadata elements (e.g. model name, experiment name) to create the
        metadata filename.

        Parameters
        ----------
        metadata_variable : str
            the name of the metadata variable to get, as it appears in the filename.

        Returns
        -------
        dict
            dictionary containing all the arguments to pass to ``self.load_data_from_identifiers``
            required to load the desired metadata cube.
        """
        raise NotImplementedError()

    @staticmethod
    def _raise_path_error(path):
        raise ValueError("Path does not look right: {}".format(path))

    @staticmethod
    def _raise_filename_error(filename):
        raise ValueError("Filename does not look right: {}".format(filename))

    @abstractproperty
    def _variable_name_for_constraint_loading(self):
        """Variable name to use when loading an iris cube with a constraint"""

    @abstractproperty
    def _time_id(self):
        """Accessor for getting the time id (whose name varies with drs)"""

    @_time_id.setter
    def _time_id(self, value):
        """Accessor for setting the time id (whose name varies with drs)"""
        raise NotImplementedError  # pragma: no cover


class MarbleCMIP5Cube(_CMIPCube):
    """
    Subclass of ``SCMCube`` which can be used with the ``cmip5`` directory on marble.

    This directory structure is very similar, but not quite identical, to the
    recommended CMIP5 directory structure described in section 3.1 of the
    `CMIP5 Data Reference Syntax <https://cmip.llnl.gov/cmip5/docs/cmip5_data_reference_syntax_v1-00_clean.pdf>`_.
    """

    root_dir = None
    """
    str: The root directory of the database i.e. where the cube should start its path

    e.g. ``/home/users/usertim/cmip5_25x25``
    """

    activity = None
    """str: The activity for which we want to load data e.g. 'cmip5'"""

    experiment = None
    """str: The experiment for which we want to load data e.g. '1pctCO2'"""

    mip_table = None
    """str: The mip_table for which we want to load data e.g. 'Amon'"""

    variable_name = None
    """str: The variable for which we want to load data e.g. 'tas'"""

    model = None
    """str: The model for which we want to load data e.g. 'CanESM2'"""

    ensemble_member = None
    """str: The ensemble member for which we want to load data e.g. 'r1i1p1'"""

    time_period = None
    """
    str: The time period for which we want to load data

    If ``None``, this information isn't included in the filename which is useful for
    loading metadata files which don't have a relevant time period.
    """

    file_ext = None
    """str: The file extension of the data file we want to load e.g. '.nc'"""

    mip_era = "CMIP5"
    """str: The MIP era to which this cube belongs"""

    _realm_key = "modeling_realm"

    @property
    def _table_name_for_metadata_vars(self):
        return "fx"

    def process_filename(self, filename):
        """
        Cut a filename into its identifiers

        Parameters
        ----------
        filename : str
            The filename to process. Filename here means just the filename, no path
            should be included.

        Returns
        -------
        dict
            A dictionary where each key is the identifier name and each value is the value of that identifier for the input filename
        """
        filename_bits = filename.split(self.filename_bits_separator)
        if len(filename_bits) == 6:
            time_period, file_ext = splitext(filename_bits[-1])
            ensemble_member = filename_bits[-2]
        elif len(filename_bits) == 5:
            time_period = None
            ensemble_member, file_ext = splitext(filename_bits[-1])
        else:
            self._raise_filename_error(filename)

        if not file_ext:
            self._raise_filename_error(filename)

        return {
            "variable_name": filename_bits[0],
            "mip_table": filename_bits[1],
            "model": filename_bits[2],
            "experiment": filename_bits[3],
            "ensemble_member": ensemble_member,
            "time_period": time_period,
            "file_ext": file_ext,
        }

    def process_path(self, path):
        """
        Cut a path into its identifiers

        Parameters
        ----------
        path : str
            The path to process. Path here means just the path, no filename
            should be included.

        Returns
        -------
        dict
            A dictionary where each key is the identifier name and each value is the value of that identifier for the input path
        """
        path = path[:-1] if path.endswith(os.sep) else path
        dirpath_bits = path.split(os.sep)
        if (len(dirpath_bits) < 6) or any(
            [self.filename_bits_separator in d for d in dirpath_bits[-6:]]
        ):
            self._raise_path_error(path)

        root_dir = os.sep.join(dirpath_bits[:-6])
        if not root_dir:
            root_dir = "."

        return {
            "root_dir": root_dir,
            "activity": dirpath_bits[-6],
            "variable_name": dirpath_bits[-3],
            "mip_table": dirpath_bits[-4],
            "model": dirpath_bits[-2],
            "experiment": dirpath_bits[-5],
            "ensemble_member": dirpath_bits[-1],
        }

    def get_filepath_from_load_data_from_identifiers_args(self, **kwargs):
        """
        Get the full filepath of the data to load from the arguments passed to ``self.load_data_from_identifiers``.

        Full details about the identifiers are given in Section 2 of the
        `CMIP5 Data Reference Syntax <https://cmip.llnl.gov/cmip5/docs/cmip5_data_reference_syntax_v1-00_clean.pdf>`_.

        Parameters
        ----------
        kwargs : str
            Identifiers to use to load the data

        Returns
        -------
        str
            The full filepath (path and name) of the file to load.

        Raises
        ------
        AttributeError
            An input argument does not match with the cube's data reference syntax
        """
        for name, value in kwargs.items():
            setattr(self, name, value)

        return join(self.get_data_directory(), self.get_data_filename())

    def _get_data_directory(self):
        return join(
            self.root_dir,
            self.activity,
            self.experiment,
            self.mip_table,
            self.variable_name,
            self.model,
            self.ensemble_member,
        )

    def _get_data_filename(self):
        bits_to_join = [
            self.variable_name,
            self.mip_table,
            self.model,
            self.experiment,
            self.ensemble_member,
        ]
        if self.time_period is not None:
            bits_to_join.append(self.time_period)

        return self.filename_bits_separator.join(bits_to_join) + self.file_ext

    def _get_metadata_load_arguments(self, metadata_variable):
        return {
            "root_dir": self.root_dir,
            "activity": self.activity,
            "experiment": self.experiment,
            "mip_table": self.table_name_for_metadata_vars,
            "variable_name": metadata_variable,
            "model": self.model,
            "ensemble_member": "r0i0p0",
            "time_period": None,
            "file_ext": self.file_ext,
        }

    @property
    def _variable_name_for_constraint_loading(self):
        return self.variable_name

    @property
    def _time_id(self):
        return self.time_period

    @_time_id.setter
    def _time_id(self, value):
        self.time_period = value


class CMIP6Input4MIPsCube(_CMIPCube):
    """
    Subclass of ``SCMCube`` which can be used with CMIP6 input4MIPs data

    The data must match the CMIP6 Forcing Datasets Summary, specifically the
    `Forcing Dataset Specifications <https://docs.google.com/document/d/1pU9IiJvPJwRvIgVaSDdJ4O0Jeorv_2ekEtted34K9cA/edit#heading=h.cn9f7982ycw6>`_.
    """

    root_dir = None
    """
    str: The root directory of the database i.e. where the cube should start its
            path

    e.g. ``/home/users/usertim/cmip6input``.
    """

    activity_id = None
    """
    str: The activity_id for which we want to load data.

    For these cubes, this will almost always be ``input4MIPs``.
    """

    mip_era = None
    """str: The mip_era for which we want to load data e.g. ``CMIP6``"""

    target_mip = None
    """str:The target_mip for which we want to load data e.g. ``ScenarioMIP``"""

    institution_id = None
    """str: The institution_id for which we want to load data e.g. ``UoM``"""

    source_id = None
    """
    str: The source_id for which we want to load data e.g.
            ``UoM-REMIND-MAGPIE-ssp585-1-2-0``

    This must include the institution_id.
    """

    realm = None
    """str: The realm for which we want to load data e.g. ``atmos``"""

    frequency = None
    """str: The frequency for which we want to load data e.g. ``yr``"""

    variable_id = None
    """str: The variable_id for which we want to load data e.g. ``mole-fraction-of-carbon-dioxide-in-air``"""

    grid_label = None
    """str: The grid_label for which we want to load data e.g. ``gr1-GMNHSH``"""

    version = None
    """str: The version for which we want to load data e.g. ``v20180427``"""

    dataset_category = None
    """str: The dataset_category for which we want to load data e.g.
            ``GHGConcentrations``"""

    time_range = None
    """
    str: The time range for which we want to load data e.g. ``2005-2100``

    If ``None``, this information isn't included in the filename which is useful
    for loading metadata files which don't have a relevant time period.
    """

    file_ext = None
    """str: The file extension of the data file we want to load e.g. ``.nc``"""

    def process_filename(self, filename):
        """
        Cut a filename into its identifiers

        Parameters
        ----------
        filename : str
            The filename to process. Filename here means just the filename, no path
            should be included.

        Returns
        -------
        dict
            A dictionary where each key is the identifier name and each value is the value of that identifier for the input filename
        """
        filename_bits = filename.split(self.filename_bits_separator)
        if len(filename_bits) == 7:
            time_range, file_ext = splitext(filename_bits[-1])
            grid_label = filename_bits[-2]
        elif len(filename_bits) == 6:
            time_range = None
            grid_label, file_ext = splitext(filename_bits[-1])
        else:
            self._raise_filename_error(filename)

        if not file_ext:
            self._raise_filename_error(filename)

        return {
            "variable_id": filename_bits[0],
            "activity_id": filename_bits[1],
            "dataset_category": filename_bits[2],
            "target_mip": filename_bits[3],
            "source_id": filename_bits[4],
            "grid_label": grid_label,
            "time_range": time_range,
            "file_ext": file_ext,
        }

    def process_path(self, path):
        """
        Cut a path into its identifiers

        Parameters
        ----------
        path : str
            The path to process. Path here means just the path, no filename
            should be included.

        Returns
        -------
        dict
            A dictionary where each key is the identifier name and each value is the value of that identifier for the input path
        """
        path = path[:-1] if path.endswith(os.sep) else path
        dirpath_bits = path.split(os.sep)
        if (len(dirpath_bits) < 10) or any(
            [self.filename_bits_separator in d for d in dirpath_bits[-10:]]
        ):
            self._raise_path_error(path)

        root_dir = os.sep.join(dirpath_bits[:-10])
        if not root_dir:
            root_dir = "."

        return {
            "root_dir": root_dir,
            "activity_id": dirpath_bits[-10],
            "mip_era": dirpath_bits[-9],
            "target_mip": dirpath_bits[-8],
            "institution_id": dirpath_bits[-7],
            "source_id": dirpath_bits[-6],
            "realm": dirpath_bits[-5],
            "frequency": dirpath_bits[-4],
            "variable_id": dirpath_bits[-3],
            "grid_label": dirpath_bits[-2],
            "version": dirpath_bits[-1],
        }

    def get_filepath_from_load_data_from_identifiers_args(self, **kwargs):
        """
        Get the full filepath of the data to load from the arguments passed to ``self.load_data_from_identifiers``.

        Full details about the meaning of the identifiers are given in the
        `Forcing Dataset Specifications <https://docs.google.com/document/d/1pU9IiJvPJwRvIgVaSDdJ4O0Jeorv_2ekEtted34K9cA/edit#heading=h.cn9f7982ycw6>`_.

        Parameters
        ----------
        kwargs : str
            Identifiers to use to load the data

        Returns
        -------
        str
            The full filepath (path and name) of the file to load.

        Raises
        ------
        AttributeError
            An input argument does not match with the cube's data reference syntax
        """
        for name, value in kwargs.items():
            setattr(self, name, value)

        # TODO: do time indicator/frequency checks too and make a new method for
        # checks so can be reused by different methods
        self._check_self_consistency()

        return join(self.get_data_directory(), self.get_data_filename())

    def _check_self_consistency(self):
        if str(self.institution_id) not in str(self.source_id):
            raise AssertionError("source_id must contain institution_id")

    def _get_metadata_load_arguments(self, metadata_variable):
        return {
            "root_dir": self.root_dir,
            "activity_id": self.activity_id,
            "mip_era": self.mip_era,
            "target_mip": self.target_mip,
            "institution_id": self.institution_id,
            "source_id": self.source_id,
            "realm": self.realm,
            "frequency": "fx",
            "variable_id": metadata_variable,
            "grid_label": self.grid_label,
            "version": self.version,
            "dataset_category": self.dataset_category,
            "time_range": None,
            "file_ext": self.file_ext,
        }

    def _get_data_filename(self):
        bits_to_join = [
            self.variable_id,
            self.activity_id,
            self.dataset_category,
            self.target_mip,
            self.source_id,
            self.grid_label,
        ]
        if self.time_range is not None:
            bits_to_join.append(self.time_range)

        return self.filename_bits_separator.join(bits_to_join) + self.file_ext

    def _get_data_directory(self):
        return join(
            self.root_dir,
            self.activity_id,
            self.mip_era,
            self.target_mip,
            self.institution_id,
            self.source_id,
            self.realm,
            self.frequency,
            self.variable_id,
            self.grid_label,
            self.version,
        )

    @property
    def _variable_name_for_constraint_loading(self):
        return self.variable_id.replace("-", "_")

    @property
    def _time_id(self):
        return self.time_range

    @_time_id.setter
    def _time_id(self, value):
        self.time_range = value


class CMIP6OutputCube(_CMIPCube):
    """
    Subclass of ``SCMCube`` which can be used with CMIP6 model output data

    The data must match the CMIP6 data reference syntax as specified in the 'File name
    template' and 'Directory structure template' sections of the
    `CMIP6 Data Reference Syntax <https://goo.gl/v1drZl>`_.
    """

    root_dir = None
    """
    str: The root directory of the database i.e. where the cube should start its
            path

    e.g. ``/home/users/usertim/cmip6_data``.
    """

    activity_id = None
    """
    str: The activity_id for which we want to load data.

    In CMIP6, this denotes the responsible MIP e.g. ``DCPP``.
    """

    mip_era = None
    """str: The mip_era for which we want to load data e.g. ``CMIP6``"""

    institution_id = None
    """str: The institution_id for which we want to load data e.g. ``CNRM-CERFACS``"""

    source_id = None
    """
    str: The source_id for which we want to load data e.g. ``CNRM-CM6-1``

    This was known as model in CMIP5.
    """

    experiment_id = None
    """str: The experiment_id for which we want to load data e.g. ``dcppA-hindcast``"""

    member_id = None
    """str: The member_id for which we want to load data e.g. ``s1960-r2i1p1f3``"""

    table_id = None
    """str: The table_id for which we want to load data. e.g. ``day``"""

    variable_id = None
    """str: The variable_id for which we want to load data e.g. ``pr``"""

    grid_label = None
    """str: The grid_label for which we want to load data e.g. ``grn``"""

    version = None
    """str: The version for which we want to load data e.g. ``v20160215``"""

    time_range = None
    """
    str: The time range for which we want to load data e.g. ``198001-198412``

    If ``None``, this information isn't included in the filename which is useful
    for loading metadata files which don't have a relevant time period.
    """

    file_ext = None
    """str: The file extension of the data file we want to load e.g. ``.nc``"""

    _scm_timeseries_id_map = {
        "variable": "variable_id",
        "climate_model": "source_id",
        "scenario": "experiment_id",
        "activity_id": "activity_id",
        "member_id": "member_id",
        "mip_era": "mip_era",
    }

    def process_filename(self, filename):
        """
        Cut a filename into its identifiers

        Parameters
        ----------
        filename : str
            The filename to process. Filename here means just the filename, no path
            should be included.

        Returns
        -------
        dict
            A dictionary where each key is the identifier name and each value is the value of that identifier for the input filename
        """
        filename_bits = filename.split(self.filename_bits_separator)
        if len(filename_bits) == 7:
            time_range, file_ext = splitext(filename_bits[-1])
            grid_label = filename_bits[-2]
        elif len(filename_bits) == 6:
            time_range = None
            grid_label, file_ext = splitext(filename_bits[-1])
        else:
            self._raise_filename_error(filename)

        if not file_ext:
            self._raise_filename_error(filename)

        return {
            "variable_id": filename_bits[0],
            "table_id": filename_bits[1],
            "source_id": filename_bits[2],
            "experiment_id": filename_bits[3],
            "member_id": filename_bits[4],
            "grid_label": grid_label,
            "time_range": time_range,
            "file_ext": file_ext,
        }

    def process_path(self, path):
        """
        Cut a path into its identifiers

        Parameters
        ----------
        path : str
            The path to process. Path here means just the path, no filename
            should be included.

        Returns
        -------
        dict
            A dictionary where each key is the identifier name and each value is the value of that identifier for the input path
        """
        path = path[:-1] if path.endswith(os.sep) else path
        dirpath_bits = path.split(os.sep)
        if (len(dirpath_bits) < 10) or any(
            [self.filename_bits_separator in d for d in dirpath_bits[-10:]]
        ):
            self._raise_path_error(path)

        root_dir = os.sep.join(dirpath_bits[:-10])
        if not root_dir:
            root_dir = "."

        return {
            "root_dir": root_dir,
            "mip_era": dirpath_bits[-10],
            "activity_id": dirpath_bits[-9],
            "institution_id": dirpath_bits[-8],
            "source_id": dirpath_bits[-7],
            "experiment_id": dirpath_bits[-6],
            "member_id": dirpath_bits[-5],
            "table_id": dirpath_bits[-4],
            "variable_id": dirpath_bits[-3],
            "grid_label": dirpath_bits[-2],
            "version": dirpath_bits[-1],
        }

    def get_filepath_from_load_data_from_identifiers_args(self, **kwargs):
        """
        Get the full filepath of the data to load from the arguments passed to ``self.load_data_from_identifiers``.

        Full details about the meaning of each identifier is given in Table 1 of the
        `CMIP6 Data Reference Syntax <https://goo.gl/v1drZl>`_.

        Parameters
        ----------
        kwargs : str
            Identifiers to use to load the data

        Returns
        -------
        str
            The full filepath (path and name) of the file to load.

        Raises
        ------
        AttributeError
            An input argument does not match with the cube's data reference syntax
        """
        for name, value in kwargs.items():
            setattr(self, name, value)

        return join(self.get_data_directory(), self.get_data_filename())

    def _get_metadata_load_arguments(self, metadata_variable):
        return {
            "root_dir": self.root_dir,
            "mip_era": self.mip_era,
            "activity_id": self.activity_id,
            "institution_id": self.institution_id,
            "source_id": self.source_id,
            "experiment_id": self.experiment_id,
            "member_id": self.member_id,
            "table_id": self.table_name_for_metadata_vars,
            "variable_id": metadata_variable,
            "grid_label": self.grid_label,
            "version": self.version,
            "time_range": None,
            "file_ext": self.file_ext,
        }

    def _get_data_filename(self):
        bits_to_join = [
            self.variable_id,
            self.table_id,
            self.source_id,
            self.experiment_id,
            self.member_id,
            self.grid_label,
        ]
        if self.time_range is not None:
            bits_to_join.append(self.time_range)

        return self.filename_bits_separator.join(bits_to_join) + self.file_ext

    def _get_data_directory(self):
        return join(
            self.root_dir,
            self.mip_era,
            self.activity_id,
            self.institution_id,
            self.source_id,
            self.experiment_id,
            self.member_id,
            self.table_id,
            self.variable_id,
            self.grid_label,
            self.version,
        )

    @property
    def _variable_name_for_constraint_loading(self):
        return self.variable_id.replace("-", "_")

    @property
    def _time_id(self):
        return self.time_range

    @_time_id.setter
    def _time_id(self, value):
        self.time_range = value
