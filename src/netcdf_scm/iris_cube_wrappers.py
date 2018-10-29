"""This module contains our wrappers of the iris cube.

These classes automate handling of a number of netCDF processing steps.
For example, finding surface land fraction files, applying masks to data and returning timeseries in key regions for simple climate models.
"""


from os.path import join, dirname, basename
import warnings
import traceback


import numpy as np
import pandas as pd
import iris
from iris.util import broadcast_to_shape
import iris.analysis.cartography
from pymagicc.io import MAGICCData


from .utils import (
    get_cube_timeseries_data,
    get_scm_cube_time_axis_in_calendar,
    assert_all_time_axes_same,
    take_lat_lon_mean,
    apply_mask,
)


class SCMCube(object):
    """Class for processing netCDF files for use in simple climate models.

    Common, shared operations are implemented here.
    However, methods like ``_get_data_directory`` raise ``NotImplementedError`` because these are always context dependent.
    Hence to use this base class, you must use a subclass of it which defines these context specific methods.
    """

    _sftlf_var = "sftlf"
    _areacella_var = "areacella"
    _lat_name = "latitude"
    _lon_name = "longitude"

    def load_data_from_path(self, filepath):
        """Load data from a path

        If you are using the ``SCMCube`` class directly, this method simply loads the
        path into an iris cube which can be accessed through ``self.cube``.

        If implemented on a subclass of ``SCMCube``, this method should:
        - use ``self.get_load_data_from_identifiers_args_from_filepath`` determine the suitable set of arguments to pass to
        ``self.load_data_from_identifiers`` from the filepath
        - load the data using ``self.load_data_from_identifiers`` as this method
        contains much better checks and helper components

        Parameters
        ----------
        filepath : str
            The filepath from which to load the data
        """
        self.cube = iris.load_cube(filepath)


    def load_data_from_identifiers(self, **kwargs):
        """Load data using key identifiers

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
            self.cube = iris.load_cube(
                self.get_filepath_from_load_data_from_identifiers_args(**kwargs),
                self.get_variable_constraint_from_load_data_from_identifiers_args(**kwargs),
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
            areacella_measure, data_dims=[self._lat_dim_number, self._lon_dim_number]
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
        """Get the iris variable constraint to use when loading data with ``self.load_data_from_identifiers``

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
            name of the data file from which this cube has been/will be loaded
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
        :obj:`pymagicc.io.MAGICCData`
            A pymagicc MAGICCData instance with the data in the ``df`` attribute and
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
        """Get the scm masks

        Returns
        -------
        dict
        """
        land_mask = self._get_land_mask(
            sftlf_cube=sftlf_cube, land_mask_threshold=land_mask_threshold
        )
        nh_mask = self._get_nh_mask()

        return {
            "World": np.full(nh_mask.shape, False),
            "World|Northern Hemisphere|Land": np.logical_or(nh_mask, land_mask),
            "World|Southern Hemisphere|Land": np.logical_or(~nh_mask, land_mask),
            "World|Northern Hemisphere|Ocean": np.logical_or(nh_mask, ~land_mask),
            "World|Southern Hemisphere|Ocean": np.logical_or(~nh_mask, ~land_mask),
            "World|Land": land_mask,
            "World|Ocean": ~land_mask,
            "World|Northern Hemisphere": nh_mask,
            "World|Southern Hemisphere": ~nh_mask,
        }

    def _get_land_mask(self, sftlf_cube=None, land_mask_threshold=50):
        """Get the land mask

        Returns
        -------
        np.ndarray
        """
        if sftlf_cube is None:
            sftlf_cube = self.get_metadata_cube(self._sftlf_var)

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
        lat_length = len(self._lat_dim.points)
        lon_length = len(self._lon_dim.points)

        dim_order = [self._lat_dim_number, self._lon_dim_number]
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

    def _convert_scm_timeseries_cubes_to_openscmdata(
        self, scm_timeseries_cubes, out_calendar=None
    ):
        data = {k: get_cube_timeseries_data(v) for k, v in scm_timeseries_cubes.items()}

        time_index, out_calendar = self._get_openscmdata_time_axis_and_calendar(
            scm_timeseries_cubes, out_calendar=out_calendar
        )

        try:
            model = self.model
            scenario = "_".join([self.activity, self.experiment, self.ensemble_member])
        except AttributeError:
            warn_msg = (
                "Could not determine appropriate model scenario combination, filling "
                "with 'unknown'"
            )
            warnings.warn(warn_msg)
            model = "unknown"
            scenario = "unknown"


        out_df = pd.DataFrame(data, index=time_index)
        out_df.columns = pd.MultiIndex.from_product(
            [
                [self.cube.standard_name],
                [self.cube.units.name],
                out_df.columns.tolist(),
                [model],
                [scenario],
            ],
            names=["variable", "unit", "region", "model", "scenario"],
        )
        out_df = out_df.unstack().reset_index().rename(
            {0: "value"}, axis="columns"
        )

        output = MAGICCData()
        output.df = out_df
        output.metadata["calendar"] = out_calendar
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

        # As we sometimes have to deal with long timeseries, we force the index to be
        # pd.Index and not pd.DatetimeIndex. We can't use DatetimeIndex because of a
        # pandas limitation, see
        # http://pandas-docs.github.io/pandas-docs-travis/timeseries.html#timestamp-limitations
        return pd.Index(time_axes[0], dtype="object", name="time"), out_calendar


class MarbleCMIP5Cube(SCMCube):
    """Subclass of `SCMCube` which can be used with the `cmip5` directory on marble

    This directory structure is very similar, but not quite identical, to the
    recommended CMIP5 directory structure described in section 3.1 of the `CMIP5 Data
    Reference Syntax
    <https://cmip.llnl.gov/cmip5/docs/cmip5_data_reference_syntax_v1-00_clean.pdf>`_.
    """

    def load_data_from_path(self, filepath):
        """Load data from a path

        Parameters
        ----------
        filepath : str
            The filepath from which to load the data
        """
        load_data_from_identifiers_args = self.get_load_data_from_identifiers_args_from_filepath(filepath)
        self.load_data_from_identifiers(**load_data_from_identifiers_args)


    def get_load_data_from_identifiers_args_from_filepath(self, filepath):
        """Get the set of identifiers to use to load data from a filepath

        Here we use the categories given in the `CMIP5 Data Reference Syntax
        <https://cmip.llnl.gov/cmip5/docs/cmip5_data_reference_syntax_v1-00_clean.pdf>`_.
        However the terminology and conventions aren't always exactly the same on
        marble so we require our custom implementation on top.

        Parameters
        ----------
        filepath : str
            The filepath from which to load the data
        """
        dirpath = dirname(filepath)
        filename = basename(filepath)
        raise NotImplementedError


    def get_filepath_from_load_data_from_identifiers_args(
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
        """Get the full filepath of the data to load from the arguments passed to ``self.load_data_from_identifiers``.

        Here we use the categories given in the `CMIP5 Data Reference Syntax
        <https://cmip.llnl.gov/cmip5/docs/cmip5_data_reference_syntax_v1-00_clean.pdf>`_.
        However the terminology isn't always exactly the same on marble.

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
        # TODO: test this switch
        if self.time_period is not None:
            bits_to_join.append(self.time_period)

        return "_".join(bits_to_join) + self.file_ext

    def get_variable_constraint_from_load_data_from_identifiers_args(
        self, variable_name="tas", **kwargs
    ):
        """Get the iris variable constraint to use when loading data with ``self.load_data_from_identifiers``

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
