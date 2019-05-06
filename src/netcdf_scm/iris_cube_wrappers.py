"""This module contains our wrappers of the iris cube.

These classes automate handling of a number of netCDF processing steps.
For example, finding surface land fraction files, applying masks to data and
returning timeseries in key regions for simple climate models.
"""
import os
from os.path import join, dirname, basename, splitext
import re
import warnings
import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser


import numpy as np
import pandas as pd
from openscm.scmdataframe import ScmDataFrame

try:
    import iris
    from iris.util import broadcast_to_shape, unify_time_units
    import iris.analysis.cartography
    import iris.experimental.equalise_cubes
    from iris.exceptions import CoordinateNotFoundError
    import cftime
    import cf_units
except ModuleNotFoundError:
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()


from .utils import (
    get_cube_timeseries_data,
    get_scm_cube_time_axis_in_calendar,
    assert_all_time_axes_same,
    take_lat_lon_mean,
    apply_mask,
    unify_lat_lon,
)


class SCMCube(object):
    """Class for processing netCDF files for use in simple climate models.

    Common, shared operations are implemented here.
    However, methods like ``_get_data_directory`` raise ``NotImplementedError``
    because these are always context dependent.
    Hence to use this base class, you must use a subclass of it which defines these
    context specific methods.
    """

    sftlf_var = "sftlf"
    """str: The name of the variable associated with the land-surface fraction in each gridbox.

    If required, this is used when looking for the land-surface fraction file which
    belongs to a given data file. For example, if our data file is
    ``tas_Amon_HadCM3_rcp45_r1i1p1_200601.nc`` then ``sftlf_var`` can be used to work
    out the name of the associated land-surface fraction file. In some cases, it might
    be as simple as replacing ``tas`` with the value of ``sftlf_var``.
    """

    areacella_var = "areacella"
    """str: The name of the variable associated with the area of each gridbox.

    If required, this is used to determine the area of each cell in a data file. For
    example, if our data file is ``tas_Amon_HadCM3_rcp45_r1i1p1_200601.nc`` then
    ``areacella_var`` can be used to work  out the name of the associated cell area
    file. In some cases, it might be as simple as replacing ``tas`` with the value of
    ``areacella_var``.
    """

    time_name = "time"
    """str: The expected name of the time co-ordinate in data."""

    lat_name = "latitude"
    """str: The expected name of the latitude co-ordinate in data."""

    lon_name = "longitude"
    """str: The expected name of the longitude co-ordinate in data."""

    time_period_separator = "-"
    """str: Character used to separate time period strings in the time period indicator in filenames.

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

    @property
    def time_period_regex(self):
        """:obj:`_sre.SRE_Pattern`: Regular expression which captures the timeseries identifier in input data files.

        For help on regular expressions, see :ref:`regular expressions <regular-expressions>`.
        """

        if self._time_period_regex is None:
            self._time_period_regex = re.compile(
                r".*_((\d*)" + re.escape(self.time_period_separator) + r"?(\d*)?).*"
            )
        return self._time_period_regex

    @property
    def timestamp_definitions(self):
        """dict: Definition of valid timestamp information and corresponding key values.

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
    def lon_dim(self):
        """:obj:`iris.coords.DimCoord` The longitude dimension of the data."""
        return self.cube.coord(self.lon_name)

    @property
    def lon_dim_number(self):
        """int: The index which corresponds to the longitude dimension.

        e.g. if longitude is the third dimension of the data, then
        ``self.lon_dim_number`` will be ``2`` (Python is zero-indexed).
        """
        return self.cube.coord_dims(self.lon_name)[0]

    @property
    def lat_dim(self):
        """:obj:`iris.coords.DimCoord` The latitude dimension of the data."""
        return self.cube.coord(self.lat_name)

    @property
    def lat_dim_number(self):
        """int: The index which corresponds to the latitude dimension.

        e.g. if latitude is the first dimension of the data, then
        ``self.lat_dim_number`` will be ``0`` (Python is zero-indexed).
        """
        return self.cube.coord_dims(self.lat_name)[0]

    @property
    def time_dim(self):
        """:obj:`iris.coords.DimCoord` The time dimension of the data."""
        return self.cube.coord(self.time_name)

    @property
    def time_dim_number(self):
        """int: The index which corresponds to the time dimension.

        e.g. if time is the first dimension of the data, then
        ``self.time_dim_number`` will be ``0`` (Python is zero-indexed).
        """
        return self.cube.coord_dims(self.time_name)[0]

    def _load_cube(self, filepath, constraint=None):
        self.cube = iris.load_cube(filepath, constraint=constraint)
        self._check_cube()

    def _check_cube(self):
        try:
            time_dim = self.time_dim
            gregorian = time_dim.units.calendar == "gregorian"
            year_zero = str(time_dim.units).startswith("days since 0-1-1")
        except CoordinateNotFoundError:
            gregorian = False
            year_zero = False

        if gregorian and year_zero:
            warn_msg = (
                "Your calendar is gregorian yet has units of 'days since 0-1-1'. "
                "We rectify this by removing all data before year 1 and changing the "
                "units to 'days since 1-1-1'. If you want other behaviour, you will "
                "need to use another package."
            )
            warnings.warn(warn_msg)
            self._adjust_gregorian_year_zero_units()

    def _adjust_gregorian_year_zero_units(self):
        year_zero_cube = self.cube.copy()
        year_zero_cube_time_dim = self.time_dim

        gregorian_year_zero_cube = (
            year_zero_cube_time_dim.units.calendar == "gregorian"
        ) and str(year_zero_cube_time_dim.units).startswith("days since 0-1-1")
        assert gregorian_year_zero_cube, "This function is not setup for other cases"

        new_unit_str = "days since 1-1-1"
        # converting with the new units means we're actually converting with the wrong
        # units, we use this variable to track how many years to shift back to get the
        # right time axis again
        new_units_shift = 1
        new_time_dim_unit = cf_units.Unit(
            new_unit_str, calendar=year_zero_cube_time_dim.units.calendar
        )

        tmp_time_dim = year_zero_cube_time_dim.copy()
        tmp_time_dim.units = new_time_dim_unit
        tmp_cube = iris.cube.Cube(year_zero_cube.data)
        for i, coord in enumerate(year_zero_cube.dim_coords):
            if coord.standard_name == "time":
                tmp_cube.add_dim_coord(tmp_time_dim, i)
            else:
                tmp_cube.add_dim_coord(coord, i)

        years_to_bin = 1
        first_valid_year = years_to_bin + new_units_shift

        def check_usable_data(cell):
            return first_valid_year <= cell.point.year

        usable_cube = tmp_cube.extract(iris.Constraint(time=check_usable_data))
        usable_data = usable_cube.data

        tmp_time_dim = usable_cube.coord(self.time_name)
        tmp_time = cftime.num2date(
            tmp_time_dim.points, new_unit_str, tmp_time_dim.units.calendar
        )
        # TODO: move to utils
        tmp_time = np.array([datetime(*v.timetuple()[:6]) for v in tmp_time])
        # undo the shift to new units
        usable_time = cf_units.date2num(
            tmp_time - relativedelta(years=new_units_shift),
            year_zero_cube_time_dim.units.name,
            year_zero_cube_time_dim.units.calendar,
        )
        usable_time_unit = cf_units.Unit(
            year_zero_cube_time_dim.units.name,
            calendar=year_zero_cube_time_dim.units.calendar,
        )
        usable_time_dim = iris.coords.DimCoord(
            usable_time,
            standard_name=year_zero_cube_time_dim.standard_name,
            long_name=year_zero_cube_time_dim.long_name,
            var_name=year_zero_cube_time_dim.var_name,
            units=usable_time_unit,
        )

        self.cube = iris.cube.Cube(usable_data)
        for i, coord in enumerate(usable_cube.dim_coords):
            if coord.standard_name == "time":
                self.cube.add_dim_coord(usable_time_dim, i)
            else:
                self.cube.add_dim_coord(coord, i)

        # hard coding as making this list dynamically is super hard as there's so many
        # edge cases to cover
        attributes_to_copy = [
            "attributes",
            "cell_methods",
            "units",
            "var_name",
            "standard_name",
            "name",
            "metadata",
            "long_name",
        ]
        for att in attributes_to_copy:
            setattr(self.cube, att, getattr(year_zero_cube, att))

    def load_data_from_path(self, filepath):
        """Load data from a path.

        If you are using the ``SCMCube`` class directly, this method simply loads the
        path into an iris cube which can be accessed through ``self.cube``.

        If implemented on a subclass of ``SCMCube``, this method should:

        - use ``self.get_load_data_from_identifiers_args_from_filepath`` to determine the suitable set of arguments to pass to ``self.load_data_from_identifiers`` from the filepath
        - load the data using ``self.load_data_from_identifiers`` as this method contains much better checks and helper components

        Parameters
        ----------
        filepath : str
            The filepath from which to load the data.
        """
        self._load_cube(filepath)

    def get_load_data_from_identifiers_args_from_filepath(self, filepath=None):
        """Get the set of identifiers to use to load data from a filepath.

        Parameters
        ----------
        filepath : str
            The filepath from which to load the data.

        Returns
        -------
        dict
            Set of arguments which can be passed to
            ``self.load_data_from_identifiers`` to load the data in the filepath.
        """
        raise NotImplementedError()

    def load_data_in_directory(self, directory=None):
        """Load data in a directory.

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

        Raises
        ------
        ValueError
            If the files in the directory are not from the same run (i.e. their filenames are not identical except for the timestamp) or if the files don't form a continuous timeseries.
        """
        self._load_and_concatenate_files_in_directory(directory)

    def _load_and_concatenate_files_in_directory(self, directory):
        self._check_data_names_in_same_directory(directory)

        # we use a loop here to make the most of finding missing data like
        # land-surface fraction and cellarea, something iris can't automatically do
        loaded_cubes = []
        for f in sorted(os.listdir(directory)):
            self.load_data_from_path(join(directory, f))
            loaded_cubes.append(self.cube)

        loaded_cubes = iris.cube.CubeList(loaded_cubes)

        unify_time_units(loaded_cubes)
        unify_lat_lon(loaded_cubes)
        iris.experimental.equalise_cubes.equalise_attributes(loaded_cubes)

        self.cube = loaded_cubes.concatenate_cube()

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
            assert re.match(base_regexp, found_file), assertion_error_msg

            file_timestamp_bits = self._get_timestamp_bits_from_filename(found_file)
            end_time_prev = datetime.strptime(
                file_timestamp_bits_prev["timeend_str"], time_format
            )
            start_time = datetime.strptime(
                file_timestamp_bits["timestart_str"], time_format
            )

            assert (
                relativedelta(start_time, end_time_prev) == expected_timestep
            ), assertion_error_msg

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

    def load_data_from_identifiers(self, **kwargs):
        """Load data using key identifiers.

        The identifiers are used to determine the path of the file to load. The file
        is then loaded into an iris cube which can be accessed through ``self.cube``.

        Parameters
        ----------
        **kwargs
            Arguments which can then be processed by
            ``self.get_filepath_from_load_data_from_identifiers_args`` and
            ``self.get_variable_constraint_from_load_data_from_identifiers_args`` to determine the full
            filepath of the file to load and the variable constraint to use.
        """
        with warnings.catch_warnings(record=True) as w:
            self._load_cube(
                self.get_filepath_from_load_data_from_identifiers_args(**kwargs),
                self.get_variable_constraint_from_load_data_from_identifiers_args(
                    **kwargs
                ),
            )

        if w:
            self._process_load_data_from_identifiers_warnings(w)

    def _process_load_data_from_identifiers_warnings(self, w):
        area_cell_warn = "Missing CF-netCDF measure variable 'areacella'"
        for warn in w:
            if area_cell_warn in str(warn.message):
                try:
                    self._add_areacella_measure()
                except Exception:
                    custom_warn = (
                        "Tried to add areacella cube, failed as shown:\n"
                        + traceback.format_exc()
                    )
                    warnings.warn(custom_warn)
                    warn_message = "\n\nareacella warning:\n" + str(warn.message)
                    warnings.warn(warn_message)
            else:
                warnings.warn(warn.message)

    def _add_areacella_measure(self):
        areacella_cube = self.get_metadata_cube(self.areacella_var).cube
        areacella_measure = iris.coords.CellMeasure(
            areacella_cube.data,
            standard_name=areacella_cube.standard_name,
            long_name=areacella_cube.long_name,
            var_name=areacella_cube.var_name,
            units=areacella_cube.units,
            attributes=areacella_cube.attributes,
            measure="area",
        )
        self.cube.add_cell_measure(
            areacella_measure, data_dims=[self.lat_dim_number, self.lon_dim_number]
        )

    def get_filepath_from_load_data_from_identifiers_args(self, **kwargs):
        """Get the full filepath of the data to load from the arguments passed to ``self.load_data_from_identifiers``.

        This function should, in most cases, call ``self._get_data_directory`` and
        ``self._get_data_filename``.

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
        raise NotImplementedError()

    def get_variable_constraint_from_load_data_from_identifiers_args(self, **kwargs):
        """Get the iris variable constraint to use when loading data with ``self.load_data_from_identifiers``.

        Parameters
        ----------
        **kwargs
            Arguments, initially passed to ``self.load_data_from_identifiers`` from which the full
            filepath of the file to load should be determined.

        Returns
        -------
        :obj:`iris.Constraint`
            constraint to use which ensures that only the variable of interest is loaded.
        """
        raise NotImplementedError()

    def _get_data_directory(self):
        """Get the path to a data file from self's attributes.

        This can take multiple forms, it may just return a previously set
        filepath attribute or it could combine a number of different metadata
        elements (e.g. model name, experiment name) to create the data path.

        Returns
        -------
        str
            path to the data file from which this cube has been/will be loaded
        """
        raise NotImplementedError()

    def _get_data_filename(self):
        """Get the name of a data file from self's attributes.

        This can take multiple forms, it may just return a previously set
        filename attribute or it could combine a number of different metadata
        elements (e.g. model name, experiment name) to create the data name.

        Returns
        -------
        str
            name of the data file from which this cube has been/will be loaded.
        """
        raise NotImplementedError()

    def get_metadata_cube(self, metadata_variable):
        """Load a metadata cube from self's attributes.

        Parameters
        ----------
        metadata_variable : str
            the name of the metadata variable to get, as it appears in the filename.

        Returns
        -------
        :obj:`type(self)`
            instance of self which has been loaded from the file containing the metadata variable of interest.
        """
        load_args = self._get_metadata_load_arguments(metadata_variable)

        metadata_cube = type(self)()
        metadata_cube.load_data_from_identifiers(**load_args)

        return metadata_cube

    def _get_metadata_load_arguments(self, metadata_variable):
        """Get the arguments to load a metadata file from self's attributes.

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

    def get_scm_timeseries(
        self, sftlf_cube=None, land_mask_threshold=50, areacella_scmcube=None
    ):
        """Get SCM relevant timeseries from ``self``.

        Parameters
        ----------
        sftlf_cube : :obj:`SCMCube`, optional
            land surface fraction data which is used to determine whether a given
            gridbox is land or ocean. If ``None``, we try to load the land surface fraction automatically.

        land_mask_threshold : float, optional
            if the surface land fraction in a grid box is greater than
            ``land_mask_threshold``, it is considered to be a land grid box.

        areacella_scmcube : :obj:`SCMCube`, optional
            cell area data which is used to take the latitude-longitude mean of the
            cube's data. If ``None``, we try to load this data automatically and if
            that fails we fall back onto ``iris.analysis.cartography.area_weights``.

        Returns
        -------
        :obj:`openscm.io.ScmDataFrame`
            An OpenSCM DataFrame instance with the data in the ``data`` attribute and
            metadata in the ``metadata`` attribute.
        """
        scm_timeseries_cubes = self.get_scm_timeseries_cubes(
            sftlf_cube=sftlf_cube,
            land_mask_threshold=land_mask_threshold,
            areacella_scmcube=areacella_scmcube,
        )

        return self._convert_scm_timeseries_cubes_to_openscmdata(scm_timeseries_cubes)

    def get_scm_timeseries_cubes(
        self, sftlf_cube=None, land_mask_threshold=50, areacella_scmcube=None
    ):
        """Get SCM relevant cubes from the ``self``.

        Parameters
        ----------
        sftlf_cube : :obj:`SCMCube`, optional
            land surface fraction data which is used to determine whether a given
            gridbox is land or ocean. If ``None``, we try to load the land surface fraction automatically.

        land_mask_threshold : float, optional
            if the surface land fraction in a grid box is greater than
            ``land_mask_threshold``, it is considered to be a land grid box.

        areacella_scmcube : :obj:`SCMCube`, optional
            cell area data which is used to take the latitude-longitude mean of the
            cube's data. If ``None``, we try to load this data automatically and if
            that fails we fall back onto ``iris.analysis.cartography.area_weights``.

        Returns
        -------
        dict
            Cubes, with latitude-longitude mean data as appropriate for each of the
            SCM relevant regions.
        """
        area_weights = self._get_area_weights(areacella_scmcube=areacella_scmcube)
        scm_cubes = self.get_scm_cubes(
            sftlf_cube=sftlf_cube, land_mask_threshold=land_mask_threshold
        )
        return {
            k: take_lat_lon_mean(scm_cube, area_weights)
            for k, scm_cube in scm_cubes.items()
        }

    def get_scm_cubes(self, sftlf_cube=None, land_mask_threshold=50):
        """Get SCM relevant cubes from the ``self``.

        Parameters
        ----------
        sftlf_cube : :obj:`SCMCube`, optional
            land surface fraction data which is used to determine whether a given
            gridbox is land or ocean. If ``None``, we try to load the land surface fraction automatically.

        land_mask_threshold : float, optional
            if the surface land fraction in a grid box is greater than
            ``land_mask_threshold``, it is considered to be a land grid box.

        Returns
        -------
        dict
            Cubes, with data masked as appropriate for each of the SCM relevant
            regions.
        """
        scm_masks = self._get_scm_masks(
            sftlf_cube=sftlf_cube, land_mask_threshold=land_mask_threshold
        )

        return {k: apply_mask(self, mask) for k, mask in scm_masks.items()}

    def _get_scm_masks(self, sftlf_cube=None, land_mask_threshold=50):
        """Get the scm masks.

        Returns
        -------
        dict
        """
        nh_mask = self._get_nh_mask()
        hemisphere_masks = {
            "World": np.full(nh_mask.shape, False),
            "World|Northern Hemisphere": nh_mask,
            "World|Southern Hemisphere": ~nh_mask,
        }

        try:
            land_mask = self._get_land_mask(
                sftlf_cube=sftlf_cube, land_mask_threshold=land_mask_threshold
            )
            land_masks = {
                "World|Northern Hemisphere|Land": np.logical_or(nh_mask, land_mask),
                "World|Southern Hemisphere|Land": np.logical_or(~nh_mask, land_mask),
                "World|Northern Hemisphere|Ocean": np.logical_or(nh_mask, ~land_mask),
                "World|Southern Hemisphere|Ocean": np.logical_or(~nh_mask, ~land_mask),
                "World|Land": land_mask,
                "World|Ocean": ~land_mask,
            }

            return {**hemisphere_masks, **land_masks}

        except OSError:
            warn_msg = (
                "Land surface fraction (sftlf) data not available, only returning "
                "global and hemispheric masks."
            )
            warnings.warn(warn_msg)
            return hemisphere_masks

    def _get_land_mask(self, sftlf_cube=None, land_mask_threshold=50):
        """Get the land mask.

        Returns
        -------
        np.ndarray
        """
        if sftlf_cube is None:
            sftlf_cube = self.get_metadata_cube(self.sftlf_var)

        if not isinstance(sftlf_cube, SCMCube):
            raise TypeError("sftlf_cube must be an SCMCube instance")

        sftlf_data = sftlf_cube.cube.data

        land_mask = np.where(
            sftlf_data > land_mask_threshold,
            False,  # where it's land, return False i.e. don't mask
            True,  # otherwise True
        )

        return self._broadcast_onto_self_lat_lon_grid(land_mask)

    def _broadcast_onto_self_lat_lon_grid(self, array_in):
        """Broadcast an array onto the latitude-longitude grid of ``self``.

        Here, broadcasting means taking the array and 'duplicating' it so that it
        has the same number of dimensions as the cube's underlying data. For example,
        if our cube has a time dimension of length 3, a latitude dimension of length 4
        and a longitude dimension of length 2 then if we are given in a 4x2 array, we
        broadcast this onto a 3x4x2 array where each slice in the broadcasted array's
        time dimension is identical to the input array.
        """
        lat_length = len(self.lat_dim.points)
        lon_length = len(self.lon_dim.points)

        dim_order = [self.lat_dim_number, self.lon_dim_number]
        base_shape = (lat_length, lon_length)
        if array_in.shape != base_shape:
            array_in = np.transpose(array_in)

        shape_assert_msg = (
            "the sftlf_cube data must be the same shape as the "
            "cube's longitude-latitude grid"
        )
        assert array_in.shape == base_shape, shape_assert_msg

        return broadcast_to_shape(array_in, self.cube.shape, dim_order)

    def _get_nh_mask(self):
        mask_nh_lat = np.array(
            [cell < 0 for cell in self.cube.coord(self.lat_name).cells()]
        )
        mask_all_lon = np.full(self.cube.coord(self.lon_name).points.shape, False)

        # Here we make a grid which we can use as a mask. We have to use all
        # of these nots so that our product (which uses AND logic) gives us
        # False in the NH and True in the SH (another way to think of this is
        # that we have to flip everything so False goes to True and True goes
        # to False, do all our operations with AND logic, then flip everything
        # back).
        mask_nh = ~np.outer(~mask_nh_lat, ~mask_all_lon)

        return self._broadcast_onto_self_lat_lon_grid(mask_nh)

    def _get_area_weights(self, areacella_scmcube=None):
        use_self_area_weights = True
        if areacella_scmcube is None:
            with warnings.catch_warnings(record=True) as w:
                areacella_scmcube = self._get_areacella_scmcube()
            if w:
                use_self_area_weights = False
                for warn in w:
                    warnings.warn(warn.message)

        if use_self_area_weights:
            try:
                areacella_cube = areacella_scmcube.cube
                return self._broadcast_onto_self_lat_lon_grid(areacella_cube.data)
            except AssertionError as exc:
                warnings.warn(str(exc))

        warnings.warn(
            "Couldn't find/use areacella_cube, falling back to iris.analysis.cartography.area_weights"
        )
        return iris.analysis.cartography.area_weights(self.cube)

    def _get_areacella_scmcube(self):
        try:
            areacella_scmcube = self.get_metadata_cube(self.areacella_var)
            if not isinstance(areacella_scmcube.cube, iris.cube.Cube):
                warnings.warn(
                    "areacella cube which was found has cube attribute which isn't an iris cube"
                )
            else:
                return areacella_scmcube
        except iris.exceptions.ConstraintMismatchError as exc:
            warnings.warn(str(exc))
        except AttributeError as exc:
            warnings.warn(str(exc))
        except OSError as exc:
            warnings.warn(str(exc))
        except NotImplementedError as exc:
            warnings.warn(str(exc))

    def _convert_scm_timeseries_cubes_to_openscmdata(
        self, scm_timeseries_cubes, out_calendar=None
    ):
        # could probably just use iris.pandas.to_series() here..?
        data = {k: get_cube_timeseries_data(v) for k, v in scm_timeseries_cubes.items()}

        time_index, out_calendar = self._get_openscmdata_time_axis_and_calendar(
            scm_timeseries_cubes, out_calendar=out_calendar
        )

        climate_model, scenario = self._get_climate_model_scenario()

        out_df = pd.DataFrame(data, index=time_index)
        out_df.columns = pd.MultiIndex.from_product(
            [
                [self.cube.standard_name],
                [self.cube.units.name],
                out_df.columns.tolist(),
                [climate_model],
                [scenario],
                ["unspecified"],
            ],
            names=["variable", "unit", "region", "climate_model", "scenario", "model"],
        )
        out_df = out_df.unstack().reset_index().rename({0: "value"}, axis="columns")
        output = ScmDataFrame(out_df)
        try:
            output.metadata["calendar"] = out_calendar
        except AttributeError:
            output.metadata = {"calendar": out_calendar}

        return output

    def _get_climate_model_scenario(self):
        try:
            climate_model = self.model
            scenario = "_".join([self.activity, self.experiment, self.ensemble_member])
        except AttributeError:
            warn_msg = (
                "Could not determine appropriate climate_model scenario combination, "
                "filling with 'unspecified'"
            )
            warnings.warn(warn_msg)
            climate_model = "unspecified"
            scenario = "unspecified"

        return climate_model, scenario

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
            time_axis = np.array([parser.parse(x.strftime()) for x in time_axis])
        elif isinstance(time_axis[0], datetime):
            pass

        # As we sometimes have to deal with long timeseries, we force the index to be
        # pd.Index and not pd.DatetimeIndex. We can't use DatetimeIndex because of a
        # pandas limitation, see
        # http://pandas-docs.github.io/pandas-docs-travis/timeseries.html#timestamp-limitations
        return pd.Index(time_axis, dtype="object", name="time"), out_calendar

    def _check_time_period_valid(self, time_period_str):
        """Check that a time_period identifier string is valid.

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
            if not start_date < end_date:
                self._raise_time_period_invalid_error(time_period_str)

    def _raise_time_period_invalid_error(self, time_period_str):
        message = "Your time_period indicator ({}) does not look right".format(
            time_period_str
        )
        raise ValueError(message)


class _CMIPCube(SCMCube):
    def load_data_from_path(self, filepath):
        """Load data from a path.

        Parameters
        ----------
        filepath : str
            The filepath from which to load the data.
        """
        load_data_from_identifiers_args = self.get_load_data_from_identifiers_args_from_filepath(
            filepath
        )
        self.load_data_from_identifiers(**load_data_from_identifiers_args)

    def _load_and_concatenate_files_in_directory(self, directory):
        super()._load_and_concatenate_files_in_directory(directory)
        self._add_time_period_from_files_in_directory(directory)

    def _add_time_period_from_files_in_directory(self, directory):
        self._check_data_names_in_same_directory(directory)

        loaded_files = sorted(os.listdir(directory))
        strt = self._get_timestamp_bits_from_filename(loaded_files[0])["timestart_str"]
        end = self._get_timestamp_bits_from_filename(loaded_files[-1])["timeend_str"]
        self.time_period = self.time_period_separator.join([strt, end])

    def get_load_data_from_identifiers_args_from_filepath(self, filepath):
        """Get the set of identifiers to use to load data from a filepath.

        Parameters
        ----------
        filepath : str
            The filepath from which to load the data.

        Returns
        -------
        dict
            Set of arguments which can be passed to
            ``self.load_data_from_identifiers`` to load the data in the filepath.
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

    def _raise_path_error(self, path):
        raise ValueError("Path does not look right: {}".format(path))

    def _raise_filename_error(self, filename):
        raise ValueError("Filename does not look right: {}".format(filename))


class MarbleCMIP5Cube(_CMIPCube):
    """Subclass of ``SCMCube`` which can be used with the ``cmip5`` directory on marble.

    This directory structure is very similar, but not quite identical, to the
    recommended CMIP5 directory structure described in section 3.1 of the
    `CMIP5 Data Reference Syntax <https://cmip.llnl.gov/cmip5/docs/cmip5_data_reference_syntax_v1-00_clean.pdf>`_.
    """

    def process_filename(self, filename):
        """Cut a filename into its identifiers

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
        filename_bits = filename.split("_")
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
            "modeling_realm": filename_bits[1],
            "model": filename_bits[2],
            "experiment": filename_bits[3],
            "ensemble_member": ensemble_member,
            "time_period": time_period,
            "file_ext": file_ext,
        }

    def process_path(self, path):
        """Cut a path into its identifiers

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
        if (len(dirpath_bits) < 6) or any(["_" in d for d in dirpath_bits[-6:]]):
            self._raise_path_error(path)

        root_dir = os.sep.join(dirpath_bits[:-6])
        if not root_dir:
            root_dir = "."

        return {
            "root_dir": root_dir,
            "activity": dirpath_bits[-6],
            "variable_name": dirpath_bits[-3],
            "modeling_realm": dirpath_bits[-4],
            "model": dirpath_bits[-2],
            "experiment": dirpath_bits[-5],
            "ensemble_member": dirpath_bits[-1],
        }

    def get_filepath_from_load_data_from_identifiers_args(
        self,
        root_dir=".",
        activity="activity",
        experiment="experiment",
        modeling_realm="modeling-realm",
        variable_name="variable-name",
        model="model",
        ensemble_member="ensemble-member",
        time_period=None,
        file_ext=".nc",
    ):
        """Get the full filepath of the data to load from the arguments passed to ``self.load_data_from_identifiers``.

        Full details about the identifiers are given in Section 2 of the
        `CMIP5 Data Reference Syntax <https://cmip.llnl.gov/cmip5/docs/cmip5_data_reference_syntax_v1-00_clean.pdf>`_.

        Parameters
        ----------
        root_dir : str, optional
            The root directory of the database i.e. where the cube should start its
            path from e.g. ``/home/users/usertim/cmip5_25x25``.

        activity : str, optional
            The activity for which we want to load data e.g. ``cmip5``.

        experiment : str, optional
            The experiment for which we want to load data e.g. ``1pctCO2``.

        modeling_realm : str, optional
            The modeling_realm for which we want to load data e.g. ``Amon``.

        variable_name : str, optional
            The variable for which we want to load data e.g. ``variable_name``.

        model : str, optional
            The model for which we want to load data ``CanESM2``.

        ensemble_member : str, optional
            The ensemble member for which we want to load data ``r1i1p1``.

        time_period : str, optional
            The time period for which we want to load data e.g. ``1850-2000``.
            If ``None``, this information isn't included in the filename which is
            useful for loading metadata files which don't have a relevant time period.

        file_ext : str, optional
            The file extension of the data file we want to load e.g. ``.nc``.

        Returns
        -------
        str
            The full filepath (path and name) of the file to load.
        """
        inargs = locals()
        del inargs["self"]
        # if the step above ever gets more complicated, use the solution here
        # http://kbyanc.blogspot.com/2007/07/python-aggregating-function-arguments.html

        for name, value in inargs.items():
            setattr(self, name, value)

        return join(self._get_data_directory(), self._get_data_filename())

    def _get_data_directory(self):
        return join(
            self.root_dir,
            self.activity,
            self.experiment,
            self.modeling_realm,
            self.variable_name,
            self.model,
            self.ensemble_member,
        )

    def _get_data_filename(self):
        bits_to_join = [
            self.variable_name,
            self.modeling_realm,
            self.model,
            self.experiment,
            self.ensemble_member,
        ]
        if self.time_period is not None:
            bits_to_join.append(self.time_period)

        return "_".join(bits_to_join) + self.file_ext

    def get_variable_constraint_from_load_data_from_identifiers_args(
        self, variable_name="tas", **kwargs
    ):
        """Get the iris variable constraint to use when loading data with ``self.load_data_from_identifiers``.

        Parameters
        ----------
        root_dir : str, optional
            The root directory of the database i.e. where the cube should start its
            path from e.g. ``/home/users/usertim/cmip5_25x25``.

        activity : str, optional
            The activity for which we want to load data.

        experiment : str, optional
            The experiment for which we want to load data.

        modeling_realm : str, optional
            The modeling_realm for which we want to load data.

        variable_name : str, optional
            The variable for which we want to load data.

        model : str, optional
            The model for which we want to load data.

        ensemble_member : str, optional
            The ensemble member for which we want to load data.

        time_period : str, optional
            The time period for which we want to load data. If ``None``, this
            information isn't included in the filename which is useful for loading
            metadata files which don't have a relevant time period.

        file_ext : str, optional
            The file extension of the data file we want to load.

        Returns
        -------
        :obj:`iris.Constraint`
            constraint to use which ensures that only the variable of interest is loaded.
        """
        # thank you Duncan!!
        # https://github.com/SciTools/iris/issues/2107#issuecomment-246644471
        return iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str(variable_name))
        )

    def _get_metadata_load_arguments(self, metadata_variable):
        return {
            "root_dir": self.root_dir,
            "activity": self.activity,
            "experiment": self.experiment,
            "modeling_realm": "fx",
            "variable_name": metadata_variable,
            "model": self.model,
            "ensemble_member": "r0i0p0",
            "time_period": None,
            "file_ext": self.file_ext,
        }


class CMIP6Input4MIPsCube(_CMIPCube):
    """Subclass of ``SCMCube`` which can be used with CMIP6 input4MIPs data

    The data must match the CMIP6 Forcing Datasets Summary, specifically the
    `Forcing Dataset Specifications <https://docs.google.com/document/d/1pU9IiJvPJwRvIgVaSDdJ4O0Jeorv_2ekEtted34K9cA/edit#heading=h.cn9f7982ycw6>`_.
    """

    def process_filename(self, filename):
        """Cut a filename into its identifiers

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
        filename_bits = filename.split("_")
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
        """Cut a path into its identifiers

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
        if (len(dirpath_bits) < 10) or any(["_" in d for d in dirpath_bits[-10:]]):
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

    def get_filepath_from_load_data_from_identifiers_args(
        self,
        root_dir=".",
        activity_id="activity-id",
        mip_era="mip-era",
        target_mip="target-mip",
        institution_id="institution-id",
        source_id="source-id-including-institution-id",
        realm="realm",
        frequency="frequency",
        variable_id="variable-id",
        grid_label="grid-label",
        version="version",
        dataset_category="dataset-category",
        time_range=None,
        file_ext="file-ext",
    ):
        """Get the full filepath of the data to load from the arguments passed to ``self.load_data_from_identifiers``.

        Full details about the meaning of the identifiers are given in the
        `Forcing Dataset Specifications <https://docs.google.com/document/d/1pU9IiJvPJwRvIgVaSDdJ4O0Jeorv_2ekEtted34K9cA/edit#heading=h.cn9f7982ycw6>`_.

        Parameters
        ----------
        root_dir : str, optional
            The root directory of the database i.e. where the cube should start its
            path from e.g. ``/home/users/usertim/cmip5_25x25``.

        activity_id : str, optional
            The activity_id for which we want to load data. For these cubes, will
            almost always be ``input4MIPs``.

        mip_era : str, optional
            The mip_era for which we want to load data e.g. ``CMIP6``.

        target_mip : str, optional
            The target_mip for which we want to load data e.g. ``ScenarioMIP``.

        institution_id : str, optional
            The institution_id for which we want to load data e.g. ``UoM``.

        source_id : str, optional
            The source_id for which we want to load data e.g.
            ``UoM-REMIND-MAGPIE-ssp585-1-2-0``. This must include the institution_id.

        realm : str, optional
            The realm for which we want to load data e.g. ``atmos``.

        frequency : str, optional
            The frequency for which we want to load data e.g. ``yr``.

        variable_id : str, optional
            The variable_id for which we want to load data e.g.
            ``mole-fraction-of-carbon-dioxide-in-air``.

        grid_label : str, optional
            The grid_label for which we want to load data e.g. ``gr1-GMNHSH``.

        version : str, optional
            The version for which we want to load data e.g. ``v20180427``.

        dataset_category : str, optional
            The dataset_category for which we want to load data e.g.
            ``GHGConcentrations``.

        time_range : str, optional
            The time range for which we want to load data e.g. ``2005-2100``. If
            ``None``, this information isn't included in the filename which is useful
            for loading metadata files which don't have a relevant time period.

        file_ext : str, optional
            The file extension of the data file we want to load e.g. ``.nc``.

        Returns
        -------
        str
            The full filepath (path and name) of the file to load.
        """
        inargs = locals()
        del inargs["self"]
        # if the step above ever gets more complicated, use the solution here
        # http://kbyanc.blogspot.com/2007/07/python-aggregating-function-arguments.html

        for name, value in inargs.items():
            setattr(self, name, value)

        # TODO: do time indicator/frequency checks too and make a new method for checks so can be reused by different methods
        self._check_self_consistency()

        return join(self._get_data_directory(), self._get_data_filename())

    def get_variable_constraint_from_load_data_from_identifiers_args(
        self, variable_id="tas", **kwargs
    ):
        """Get the iris variable constraint to use when loading data with ``self.load_data_from_identifiers``.

        Parameters
        ----------
        root_dir : str, optional
            The root directory of the database i.e. where the cube should start its
            path from e.g. ``/home/users/usertim/cmip5_25x25``.

        activity_id : str, optional
            The activity_id for which we want to load data. For these cubes, will
            almost always be "input4MIPs".

        mip_era : str, optional
            The mip_era for which we want to load data.

        target_mip : str, optional
            The target_mip for which we want to load data.

        institution_id : str, optional
            The institution_id for which we want to load data.

        source_id : str, optional
            The source_id for which we want to load data. This must include the version and the institution_id.

        realm : str, optional
            The realm for which we want to load data.

        frequency : str, optional
            The frequency for which we want to load data.

        variable_id : str, optional
            The variable_id for which we want to load data.

        grid_label : str, optional
            The grid_label for which we want to load data.

        version : str, optional
            The version for which we want to load data.

        dataset_category : str, optional
            The dataset_category for which we want to load data.

        time_range : str, optional
            The time range for which we want to load data. If ``None``, this
            information isn't included in the filename which is useful for loading
            metadata files which don't have a relevant time period.

        file_ext : str, optional
            The file extension of the data file we want to load.

        Returns
        -------
        :obj:`iris.Constraint`
            constraint to use which ensures that only the variable of interest is loaded.
        """
        # thank you Duncan!!
        # https://github.com/SciTools/iris/issues/2107#issuecomment-246644471
        return iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str(variable_id.replace("-", "_")))
        )

    def _check_self_consistency(self):
        assert (
            self.institution_id in self.source_id
        ), "source_id must contain institution_id"

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

        return "_".join(bits_to_join) + self.file_ext

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


class CMIP6OutputCube(_CMIPCube):
    """Subclass of ``SCMCube`` which can be used with CMIP6 model output data

    The data must match the CMIP6 data reference syntax as specified in the 'File name
    template' and 'Directory structure template' sections of the
    `CMIP6 Data Reference Syntax <https://goo.gl/v1drZl>`_.
    """

    def process_filename(self, filename):
        """Cut a filename into its identifiers

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
        filename_bits = filename.split("_")
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
        """Cut a path into its identifiers

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
        if (len(dirpath_bits) < 10) or any(["_" in d for d in dirpath_bits[-10:]]):
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

    def get_filepath_from_load_data_from_identifiers_args(
        self,
        root_dir=".",
        mip_era="mip-era",
        activity_id="activity-id",
        institution_id="institution-id",
        source_id="source-id",
        experiment_id="experiment-id",
        member_id="member-id",
        table_id="table-id",
        variable_id="variable-id",
        grid_label="grid-label",
        version="version",
        time_range=None,
        file_ext="file-ext",
    ):
        """Get the full filepath of the data to load from the arguments passed to ``self.load_data_from_identifiers``.

        Full details about the meaning of each identifier is given in Table 1 of the
        `CMIP6 Data Reference Syntax <https://goo.gl/v1drZl>`_.

        Parameters
        ----------
        root_dir : str, optional
            The root directory of the database i.e. where the cube should start its
            path from e.g. ``/home/users/usertim/cmip6_data``.

        mip_era : str, optional
            The mip_era for which we want to load data e.g. ``CMIP6``.

        activity_id : str, optional
            The activity for which we want to load data e.g. ``DCPP``.

        institution_id : str, optional
            The institution for which we want to load data e.g. ``CNRM-CERFACS``

        source_id : str, optional
            The source_id for which we want to load data e.g. ``CNRM-CM6-1``. This was
            known as model in CMIP5.

        experiment_id : str, optional
            The experiment_id for which we want to load data e.g. ``dcppA-hindcast``.

        member_id : str, optional
            The member_id for which we want to load data e.g. ``s1960-r2i1p1f3``.

        table_id : str, optional
            The table_id for which we want to load data. e.g. ``day``.

        variable_id : str, optional
            The variable_id for which we want to load data e.g. ``pr``.

        grid_label : str, optional
            The grid_label for which we want to load data e.g. ``gn``.

        version : str, optional
            The version for which we want to load data e.g. ``v20160215``.

        time_range : str, optional
            The time range for which we want to load data e.g. ``198001-198412``. If
            ``None``, this information isn't included in the filename which is useful
            for loading metadata files which don't have a relevant time period.

        file_ext : str, optional
            The file extension of the data file we want to load e.g. ``.nc``.

        Returns
        -------
        str
            The full filepath (path and name) of the file to load.
        """
        inargs = locals()
        del inargs["self"]
        # if the step above ever gets more complicated, use the solution here
        # http://kbyanc.blogspot.com/2007/07/python-aggregating-function-arguments.html

        for name, value in inargs.items():
            setattr(self, name, value)

        return join(self._get_data_directory(), self._get_data_filename())

    def get_variable_constraint_from_load_data_from_identifiers_args(
        self, variable_id="tas", **kwargs
    ):
        """Get the iris variable constraint to use when loading data with ``self.load_data_from_identifiers``.

        Parameters
        ----------
        root_dir : str, optional
            The root directory of the database i.e. where the cube should start its
            path from e.g. ``/home/users/usertim/cmip5_25x25``.

        mip_era : str, optional
            The mip_era for which we want to load data.

        activity_id : str, optional
            The activity_id for which we want to load data. For these cubes, will
            almost always be "input4MIPs".

        institution_id : str, optional
            The institution_id for which we want to load data.

        source_id : str, optional
            The source_id for which we want to load data. This was known as model in
            CMIP5.

        experiment_id : str, optional
            The experiment_id for which we want to load data.

        member_id : str, optional
            The member_id for which we want to load data.

        table_id : str, optional
            The table_id for which we want to load data.

        variable_id : str, optional
            The variable_id for which we want to load data.

        grid_label : str, optional
            The grid_label for which we want to load data.

        version : str, optional
            The version for which we want to load data.

        time_range : str, optional
            The time range for which we want to load data. If ``None``, this
            information isn't included in the filename which is useful for loading
            metadata files which don't have a relevant time period.

        file_ext : str, optional
            The file extension of the data file we want to load.

        Returns
        -------
        :obj:`iris.Constraint`
            constraint to use which ensures that only the variable of interest is loaded.
        """
        # thank you Duncan!!
        # https://github.com/SciTools/iris/issues/2107#issuecomment-246644471
        return iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str(variable_id.replace("-", "_")))
        )

    def _get_metadata_load_arguments(self, metadata_variable):
        return {
            "root_dir": self.root_dir,
            "mip_era": self.mip_era,
            "activity_id": self.activity_id,
            "institution_id": self.institution_id,
            "source_id": self.source_id,
            "experiment_id": self.experiment_id,
            "member_id": "r0i0p0",
            "table_id": "fx",
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

        return "_".join(bits_to_join) + self.file_ext

    def _get_data_directory(self):
        return join(
            self.root_dir,
            self.mip_era,
            self.activity_id,
            self.institution_id,
            self.source_id,
            self.experiment_id,
            self.member_id,
            self.variable_id,
            self.grid_label,
            self.version,
        )
