from os import listdir
from os.path import join, splitext, basename, isdir
from datetime import datetime
import warnings


import numpy as np
import pandas as pd
import re
import iris
from iris.util import broadcast_to_shape
import iris.analysis.cartography
from pymagicc.io import MAGICCData


class _SCMCube(object):
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
            `self._get_file_from_load_data_args` to determine the full
            filepath of the file to load.

        # Side Effects
        - sets the `cube` attribute to the loaded iris cube.
        """
        # validate args, need to check if positional args can be switched for keyword in Python3
        # set attributes
        # deal with time period
        # load cube
        self.cube = iris.load_cube(
            self._get_file_from_load_data_args(**kwargs),
            self._get_variable_constraint_from_load_data_args(**kwargs),
        )

    def _get_file_from_load_data_args(self, **kwargs):
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

    def _get_variable_constraint_from_load_data_args(self, **kwargs):
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
        Get the name of a metadata file from self's attributes.

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
        self, sftlf_cube=None, land_mask_threshold=50, areacella_cube=None
    ):
        """

        """
        scm_timeseries_cubes = self.get_scm_timeseries_cubes(
            sftlf_cube=sftlf_cube,
            land_mask_threshold=land_mask_threshold,
            areacella_cube=areacella_cube,
        )
        # TODO: actually use openscm DataFrame as return value
        return self._convert_scm_timeseries_cubes_to_OpenSCMData(scm_timeseries_cubes)

    def get_scm_timeseries_cubes(
        self, sftlf_cube=None, land_mask_threshold=50, areacella_cube=None
    ):
        """

        """

        def take_mean(in_cube, in_mask, in_weights):
            out_cube = in_cube.copy()
            out_cube.data = np.ma.asarray(out_cube.data)
            out_cube.data.mask = in_mask

            return out_cube.collapsed(
                [self._lat_name, self._lon_name], iris.analysis.MEAN, weights=in_weights
            )

        scm_masks = self._get_scm_masks(
            sftlf_cube=sftlf_cube, land_mask_threshold=land_mask_threshold
        )
        area_weights = self._get_area_weights(self, areacella_cube=areacella_cube)

        return {
            k: take_mean(self.cube, mask, area_weights) for k, mask in scm_masks.items()
        }
        raise NotImplementedError()

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

        if isinstance(sftlf_cube, _SCMCube):
            sftlf_data = sftlf_cube.cube.data
        else:
            assert isinstance(
                sftlf_cube, np.ndarray
            ), "sftlf_cube must be a numpy.ndarray if it's not an _SCMCube instance"

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

        return broadcast_to_shape(
            array_in,
            self.cube.shape,
            dim_order,
        )

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

    def _get_area_weights(self, areacella_cube=None):
        """

        """
        use_self_area_weights = True
        if areacella_cube is None:
            try:
                areacella_cube = self.get_metadata_cube(self._areacella_var)
                if not isinstance(areacella_cube.cube, iris.cube.Cube):
                    warnings.warn("areacella cube which was found has cube attribute which isn't an iris cube")
                    use_self_area_weights = False
            except iris.exceptions.ConstraintMismatchError as exc:
                warnings.warn(str(exc))
                use_self_area_weights = False
            except AttributeError as exc:
                warnings.warn(str(exc))
                use_self_area_weights = False

        if use_self_area_weights:
            try:
                return self._broadcast_onto_self_lat_lon_grid(areacella_cube.cube.data)
            except AssertionError as exc:
                warnings.warn(str(exc))

        warnings.warn(
            "Couldn't find/use areacella_cube, falling back to iris.analysis.cartography.area_weights"
        )
        return iris.analysis.cartography.area_weights(self.cube)


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



    def _convert_scm_timeseries_cubes_to_OpenSCMData(self, scm_timeseries_cubes):
        """

        """
        raise NotImplementedError()


class MarbleCMIP5Cube(_SCMCube):
    """
    Subclass of `_SCMCube` which can be used with the `cmip5` directory on marble

    This directory structure is very similar, but not quite identical, to the
    recommended CMIP5 directory structure.
    """
