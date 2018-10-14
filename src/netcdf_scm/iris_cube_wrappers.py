from os import listdir
from os.path import join, splitext, basename, isdir
from datetime import datetime
import warnings
import traceback

import numpy as np
import pandas as pd
import re
import iris
from iris.util import broadcast_to_shape
import iris.analysis.cartography
import cf_units
from pymagicc.io import MAGICCData


class SCMCube(object):
    """
    Provides the ability to process netCDF files for use in simple climate models.

    This base class contains the most common operations. However to fully
    utilise its power you must use a subclass of it, which defines the methods
    which raise `NotImplementedError`'s in this class.
    """

    _sftlf_var = "sftlf"
    _areacella_var = "areacella"
    _lat_name = "latitude"
    _lon_name = "longitude"

    def load_data(self, **kwargs):
        """
        Load data from a netCDF file.

        # Parameters
        kwargs (dict): arguments which can then be processed by
            `self.get_file_from_load_data_args` to determine the full
            filepath of the file to load.

        # Side Effects
        - sets the `cube` attribute to the loaded iris cube.
        """
        # validate args, need to check if positional args can be switched for keyword in Python3
        # set attributes
        # deal with time period
        # load cube
        with warnings.catch_warnings(record=True) as w:
            self.cube = iris.load_cube(
                self.get_file_from_load_data_args(**kwargs),
                self.get_variable_constraint_from_load_data_args(**kwargs),
            )

        if w:
            self._process_load_data_warnings(w)

    def _process_load_data_warnings(self, w):
        area_cell_warn = "Missing CF-netCDF measure variable 'areacella'"
        for warn in w:
            if area_cell_warn in str(warn.message):
                try:
                    self._add_areacella_measure()
                except Exception as exc:
                    custom_warn = "Tried to add areacella cube, failed as shown:\n" + traceback.format_exc()
                    warnings.warn(custom_warn)
                    warn_message = "\n\nareacella warning:\n" + str(warn.message)
                    warnings.warn(warn_message)
            else:
                warnings.warn(warn.message)

    def _add_areacella_measure(self):
        areacella_cube = self.get_metadata_cube(self._areacella_var).cube
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
            areacella_measure,
            data_dims=[self._lat_dim_number, self._lon_dim_number],
        )

    def get_file_from_load_data_args(self, **kwargs):
        """
        Get the full filepath of the data to load from the arguments passed to `self.load_data`.

        This function should, in most cases, call `self._get_data_path` and
        `self._get_data_name`.

        # Parameters
        kwargs (dict): arguments, initially passed to `self.load_data` from
            which the full filepath of the file to load should be determined.

        # Returns
        fullpath (str): the full filepath (path and name) of the file to load.
        """
        raise NotImplementedError()

    def get_variable_constraint_from_load_data_args(self, **kwargs):
        """
        Get the iris variable constraint to use when loading data with `self.load_data`

        # Parameters
        kwargs (dict): arguments, initially passed to `self.load_data` from
            which the full filepath of the file to load should be determined.

        # Returns
        var_constraint (iris.Constraint): constraint to use which ensures that
            only the variable of interest is loaded.
        """
        raise NotImplementedError()

    def _get_data_path(self):
        """
        Get the path to a data file from self's attributes.

        This can take multiple forms, it may just return a previously set
        filepath attribute or it could combine a number of different metadata
        elements (e.g. model name, experiment name) to create the data path.

        # Returns
        data_path (str): path to the data file from which this cube has been/
            will be loaded
        """
        raise NotImplementedError()

    def _get_data_name(self):
        """
        Get the name of a data file from self's attributes.

        This can take multiple forms, it may just return a previously set
        filename attribute or it could combine a number of different metadata
        elements (e.g. model name, experiment name) to create the data name.

        # Returns
        data_name (str): name of the data file from which this cube has been/
            will be loaded
        """
        raise NotImplementedError()

    def get_metadata_cube(self, metadata_variable):
        """
        Load a metadata cube from self's attributes.

        # Parameters
        metadata_variable (str): the name of the metadata variable to get, as
            it appears in the filename.

        # Returns
        metadata_cube (type(self)): instance of self which has been loaded
            from the file containing the metadata variable of interest
        """
        load_args = self._get_metadata_load_arguments(metadata_variable)

        metadata_cube = type(self)()
        metadata_cube.load_data(**load_args)

        return metadata_cube

    def _get_metadata_load_arguments(self, metadata_variable):
        """
        Get the arguments to load a metadata file from self's attributes.

        This can take multiple forms, it may just return a previously set
        metada_filename attribute or it could combine a number of different
        metadata elements (e.g. model name, experiment name) to create the
        metadata filename.

        # Parameters
        metadata_variable (str): the name of the metadata variable to get, as
            it appears in the filename.

        # Returns
        load_args (dict): dictionary containing all the arguments to pass to
            `self.load_data` required to load the desired metadata cube.
        """
        raise NotImplementedError()

    def get_scm_timeseries(
        self, sftlf_cube=None, land_mask_threshold=50, areacella_scmcube=None
    ):
        """

        """
        scm_timeseries_cubes = self.get_scm_timeseries_cubes(
            sftlf_cube=sftlf_cube,
            land_mask_threshold=land_mask_threshold,
            areacella_scmcube=areacella_scmcube,
        )

        return self._convert_scm_timeseries_cubes_to_OpenSCMData(scm_timeseries_cubes)

    def get_scm_timeseries_cubes(
        self, sftlf_cube=None, land_mask_threshold=50, areacella_scmcube=None
    ):
        """

        """
        area_weights = self._get_area_weights(areacella_scmcube=areacella_scmcube)
        scm_cubes = self.get_scm_cubes(sftlf_cube=sftlf_cube, land_mask_threshold=land_mask_threshold, areacella_scmcube=areacella_scmcube)
        return {k: self.take_lat_lon_mean(cube, area_weights) for k, cube in scm_cubes.items()}

    def take_lat_lon_mean(self, in_scmcube, in_weights):
        """
        move to utils
        """
        out_cube = type(in_scmcube)()
        out_cube.cube = in_scmcube.cube.copy()
        out_cube.cube = out_cube.cube.collapsed(
            [self._lat_name, self._lon_name], iris.analysis.MEAN, weights=in_weights
        )
        return out_cube

    def get_scm_cubes(
        self, sftlf_cube=None, land_mask_threshold=50, areacella_scmcube=None
    ):
        """
        Returns SCMCubes
        """
        scm_masks = self._get_scm_masks(
            sftlf_cube=sftlf_cube, land_mask_threshold=land_mask_threshold
        )

        return {k: self.apply_mask(self, mask) for k, mask in scm_masks.items()}

    def apply_mask(self, in_scmcube, in_mask):
        """
        move to utils
        """
        out_cube = type(in_scmcube)()
        out_cube.cube = in_scmcube.cube.copy()
        out_cube.cube.data = np.ma.asarray(out_cube.cube.data)
        out_cube.cube.data.mask = in_mask

        return out_cube

    def _get_scm_masks(self, sftlf_cube=None, land_mask_threshold=50):
        """

        """
        land_mask = self._get_land_mask(
            sftlf_cube=sftlf_cube, land_mask_threshold=land_mask_threshold
        )
        nh_mask = self._get_nh_mask()

        return {
            "GLOBAL": np.full(nh_mask.shape, False),
            "NH_LAND": np.logical_or(nh_mask, land_mask),
            "SH_LAND": np.logical_or(~nh_mask, land_mask),
            "NH_OCEAN": np.logical_or(nh_mask, ~land_mask),
            "SH_OCEAN": np.logical_or(~nh_mask, ~land_mask),
        }

    def _get_land_mask(self, sftlf_cube=None, land_mask_threshold=50):
        """

        """
        if sftlf_cube is None:
            sftlf_cube = self.get_metadata_cube(self._sftlf_var)

        if isinstance(sftlf_cube, SCMCube):
            sftlf_data = sftlf_cube.cube.data
        else:
            assert isinstance(
                sftlf_cube, np.ndarray
            ), "sftlf_cube must be a numpy.ndarray if it's not an SCMCube instance"

            sftlf_data = sftlf_cube

        land_mask = np.where(
            sftlf_data > land_mask_threshold,
            False,  # where it's land, return False i.e. don't mask
            True,  # otherwise True
        )

        return self._broadcast_onto_self_lat_lon_grid(land_mask)

    def _broadcast_onto_self_lat_lon_grid(self, array_in):
        lat_length = len(self._lat_dim.points)
        lon_length = len(self._lon_dim.points)

        dim_order = [self._lat_dim_number, self._lon_dim_number]
        base_shape = (lat_length, lon_length)
        if array_in.shape != base_shape:
            array_in = np.transpose(array_in)

        shape_assert_msg = (
            "the sftlf_cube data must be the same shape as (or the transpose of) the "
            "cube's longitude-latitude grid"
        )
        assert array_in.shape == base_shape, shape_assert_msg

        return broadcast_to_shape(array_in, self.cube.shape, dim_order)

    def _get_nh_mask(self):
        """

        """
        mask_nh_lat = np.array(
            [cell < 0 for cell in self.cube.coord(self._lat_name).cells()]
        )
        mask_all_lon = np.full(self.cube.coord(self._lon_name).points.shape, False)

        # Here we make a grid which we can use as a mask. We have to use all
        # of these nots so that our product (which uses AND logic) gives us
        # False in the NH and True in the SH (another way to think of this is
        # that we have to flip everything so False goes to True and True goes
        # to False, do all our operations with AND logic, then flip everything
        # back).
        mask_nh = ~np.outer(~mask_nh_lat, ~mask_all_lon)

        return self._broadcast_onto_self_lat_lon_grid(mask_nh)

    def _get_area_weights(self, areacella_scmcube=None):
        """

        """
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
            areacella_scmcube = self.get_metadata_cube(self._areacella_var)
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

    @property
    def _lon_dim(self):
        return self.cube.coord(self._lon_name)

    @property
    def _lon_dim_number(self):
        return self.cube.coord_dims(self._lon_name)[0]

    @property
    def _lat_dim(self):
        return self.cube.coord(self._lat_name)

    @property
    def _lat_dim_number(self):
        return self.cube.coord_dims(self._lat_name)[0]

    def _convert_scm_timeseries_cubes_to_OpenSCMData(
        self, scm_timeseries_cubes, out_calendar=None
    ):
        """

        """
        if out_calendar is None:
            out_calendar = self.cube.coords("time")[0].units.calendar

        data = {k: self.get_timeseries_data(v) for k, v in scm_timeseries_cubes.items()}

        time_axes = [
            self.get_time_axis_in_calendar(scm_cube, out_calendar)
            for k, scm_cube in scm_timeseries_cubes.items()
        ]
        self._assert_all_time_axes_same(time_axes)

        # As we sometimes have to deal with long timeseries, we force the index to be
        # pd.Index and not pd.DatetimeIndex. We can't use DatetimeIndex because of a
        # pandas limitation, see
        # http://pandas-docs.github.io/pandas-docs-travis/timeseries.html#timestamp-limitations
        time_index = pd.Index(time_axes[0], dtype="object", name="Time")

        output = MAGICCData()
        output.df = pd.DataFrame(data, index=time_index)
        output.df.columns = pd.MultiIndex.from_product(
            [
                [self.cube.standard_name],
                [self.cube.units.name],
                output.df.columns.tolist(),
            ],
            names=["VARIABLE", "UNITS", "REGION"],
        )

        output.metadata["calendar"] = out_calendar
        return output

    def assert_only_time(self, scm_cube):
        """
        move to utils
        """
        assert_msg = "Should only have time coordinate here"
        assert len(scm_cube.cube.dim_coords) == 1, assert_msg
        assert scm_cube.cube.dim_coords[0].standard_name == "time"

    def get_timeseries_data(self, scm_cube):
        """
        move to utils
        """
        self.assert_only_time(scm_cube)
        return scm_cube.cube.data

    def get_time_axis_in_calendar(self, scm_cube, calendar):
        """
        move to utils
        """
        self.assert_only_time(scm_cube)
        time = scm_cube.cube.dim_coords[0]
        return cf_units.num2date(time.points, time.units.name, calendar)

    def _assert_all_time_axes_same(self, time_axes):
        """
        move to utils
        """
        for time_axis_to_check in time_axes:
            assert_msg = "all the time axes should be the same"
            np.testing.assert_array_equal(
                time_axis_to_check, time_axes[0]
            ), assert_msg


class MarbleCMIP5Cube(SCMCube):
    """
    Subclass of `SCMCube` which can be used with the `cmip5` directory on marble

    This directory structure is very similar, but not quite identical, to the
    recommended CMIP5 directory structure described in section 3.1 of the [CMIP5 Data
    Reference Syntax]
    (https://cmip.llnl.gov/cmip5/docs/cmip5_data_reference_syntax_v1-00_clean.pdf)
    """

    def get_file_from_load_data_args(
        self,
        root_dir=".",
        activity="cmip5",
        experiment="1pctCO2",
        modeling_realm="Amon",
        variable_name="tas",
        model="CanESM2",
        ensemble_member="r1i1p1",
        time_period=None,
        file_ext=None,
    ):
        """
        Get the full filepath of the data to load from the arguments passed to `self.load_data`.

        TODO: implement fancy stuff like working out time period and file extension
        TODO: rewrite Parameters
        # Parameters
        kwargs (dict): arguments, initially passed to `self.load_data` from
            which the full filepath of the file to load should be determined.

        # Returns
        fullpath (str): the full filepath (path and name) of the file to load.
        """
        # if this ever gets more complicated, use the solution here
        # http://kbyanc.blogspot.com/2007/07/python-aggregating-function-arguments.html
        inargs = locals()
        del inargs["self"]

        for name, value in inargs.items():
            setattr(self, name, value)

        return join(self._get_data_path(), self._get_data_name())

    def _get_data_path(self):
        """
        Get the path to a data file from self's attributes.

        # Returns
        data_path (str): path to the data file from which this cube has been/
            will be loaded
        """
        return join(
            self.root_dir,
            self.activity,
            self.experiment,
            self.modeling_realm,
            self.variable_name,
            self.model,
            self.ensemble_member,
        )

    def _get_data_name(self):
        """
        Get the name of a data file from self's attributes.

        # Returns
        data_name (str): name of the data file from which this cube has been/
            will be loaded
        """
        bits_to_join = [
            self.variable_name,
            self.modeling_realm,
            self.model,
            self.experiment,
            self.ensemble_member,
        ]
        # TODO: test this switch
        if self.time_period is not None:
            bits_to_join.append(self.time_period)

        return "_".join(bits_to_join) + self.file_ext

    def get_variable_constraint_from_load_data_args(
        self, variable_name="tas", **kwargs
    ):
        """
        Get the iris variable constraint to use when loading data with `self.load_data`

        TODO: rewrite Parameters
        # Parameters
        kwargs (dict): arguments, initially passed to `self.load_data` from
            which the full filepath of the file to load should be determined.

        # Returns
        var_constraint (iris.Constraint): constraint to use which ensures that
            only the variable of interest is loaded.
        """
        # thank you Duncan!!
        # https://github.com/SciTools/iris/issues/2107#issuecomment-246644471
        return iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str(variable_name))
        )

    def _get_metadata_load_arguments(self, metadata_variable):
        """
        Get the arguments to load a metadata file from self's attributes.

        This can take multiple forms, it may just return a previously set
        metada_filename attribute or it could combine a number of different
        metadata elements (e.g. model name, experiment name) to create the
        metadata filename.

        # Parameters
        metadata_variable (str): the name of the metadata variable to get, as
            it appears in the filename.

        # Returns
        load_args (dict): dictionary containing all the arguments to pass to
            `self.load_data` required to load the desired metadata cube.
        """
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
