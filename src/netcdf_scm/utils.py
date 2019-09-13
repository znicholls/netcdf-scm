"""
Utils contains a number of helpful functions for doing common cube operations.

For example, applying masks to cubes, taking latitude-longitude means and getting
timeseries from a cube as datetime values.
"""
import datetime as dt
import logging

import numpy as np
import numpy.ma as ma
from dateutil.relativedelta import relativedelta

try:
    import cftime
    import dask.array as da
    import iris
    from iris.analysis import WeightedAggregator, _build_dask_mdtol_function
    from iris.exceptions import CoordinateNotFoundError
    import cf_units

    # monkey patch iris MEAN until https://github.com/SciTools/iris/pull/3299 is merged
    iris.analysis.MEAN = WeightedAggregator(
        "mean", ma.average, lazy_func=_build_dask_mdtol_function(da.ma.average)
    )

except ModuleNotFoundError:  # pragma: no cover # emergency valve
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()

logger = logging.getLogger(__name__)


def get_cube_timeseries_data(scm_cube, realise_data=False):
    """
    Get a timeseries from a cube.

    This function only works on cubes which are on a time grid only i.e. have no other dimension coordinates.

    Parameters
    ----------
    scm_cube : :obj:`SCMCube`
        An ``SCMCube`` instance with only a 'time' dimension.

    realise_data : bool
        If ``True``, force the data to be realised before returning

    Returns
    -------
    np.ndarray
        The cube's timeseries data. If ``realise_data`` is ``False`` then a
        ``da.Array`` will be returned if the data is lazy.
    """
    _assert_only_cube_dim_coord_is_time(scm_cube)
    raw_data = scm_cube.cube.core_data()
    if isinstance(raw_data, da.Array) and realise_data:
        return raw_data.compute()
    return raw_data


def get_scm_cube_time_axis_in_calendar(scm_cube, calendar):
    """
    Get a cube's time axis in a given calendar

    Parameters
    ----------
    scm_cube : :obj:`SCMCube`
        An ``SCMCube`` instance.
    calendar : str
        The calendar to return the time axis in e.g. '365_day', 'gregorian'.

    Returns
    -------
    np.ndarray
        Array of datetimes, containing the cube's calendar.
    """
    time = scm_cube.cube.dim_coords[scm_cube.time_dim_number]
    return cf_units.num2date(time.points, time.units.name, calendar)


def _assert_only_cube_dim_coord_is_time(scm_cube):
    assert_msg = "Should only have time coordinate here"
    if len(scm_cube.cube.dim_coords) != 1:
        raise AssertionError(assert_msg)
    if scm_cube.cube.dim_coords[0].standard_name != "time":
        raise AssertionError(assert_msg)


def assert_all_time_axes_same(time_axes):
    """
    Assert all time axes in a set are the same.

    Parameters
    ----------
    time_axes : list_like of array_like
        List of time axes to compare.

    Raises
    ------
    AssertionError
        If not all time axes are the same.
    """
    for time_axis_to_check in time_axes:
        assert_msg = "all the time axes should be the same"
        try:
            np.testing.assert_array_equal(
                time_axis_to_check, time_axes[0], err_msg=assert_msg
            )
        # handle weird numpy error in case it comes up
        except AttributeError:  # pragma: no cover
            raise AssertionError(assert_msg)


def take_lat_lon_mean(in_scmcube, in_weights):
    """
    Take the latitude longitude mean of a cube with given weights

    Parameters
    ----------
    in_scmcube : :obj:`SCMCube`
        An ``SCMCube`` instance.

    in_weights : np.ndarray
        Weights to use when taking the mean.

    Returns
    -------
    :obj:`SCMCube`
        A copy of the input cube in which the data is now the latitude-longitude mean
        of the input cube's data
    """
    out_cube = type(in_scmcube)()
    if in_weights.shape != in_scmcube.cube.shape:
        in_weights = broadcast_onto_lat_lon_grid(in_scmcube, in_weights)

    out_cube.cube = in_scmcube.cube.collapsed(
        [in_scmcube.lat_name, in_scmcube.lon_name],
        iris.analysis.MEAN,
        weights=in_weights,
    )
    return out_cube


def apply_mask(in_scmcube, in_mask):
    """
    Apply a mask to an scm cube's data

    Parameters
    ----------
    in_scmcube : :obj:`SCMCube`
        An ``SCMCube`` instance.

    in_mask : np.ndarray
        The mask to apply

    Returns
    -------
    :obj:`SCMCube`
        A copy of the input cube with the mask applied to its data
    """
    out_cube = type(in_scmcube)()
    if in_scmcube.cube.has_lazy_data():
        new_data = da.ma.masked_array(data=in_scmcube.cube.lazy_data(), mask=in_mask)
    else:
        new_data = ma.masked_array(in_scmcube.cube.data, mask=in_mask)
    out_cube.cube = in_scmcube.cube.copy(data=new_data)

    return out_cube


def unify_lat_lon(cubes, rtol=10 ** -6):
    """
    Unify latitude and longitude co-ordinates of cubes in place.

    The co-ordinates will only be unified if they already match to within a given
    tolerance.

    Parameters
    ----------
    cubes : :obj:`iris.cube.CubeList`
        List of iris cubes whose latitude and longitude co-ordinates should be unified.

    rtol : float
        Maximum relative difference which can be accepted between co-ordinate values.

    Raises
    ------
    ValueError
        If the co-ordinates differ by more than relative tolerance or are not
        compatible (e.g. different shape).
    """
    ref_lats = cubes[0].coords("latitude")[0].points
    ref_lons = cubes[0].coords("longitude")[0].points
    for cube in cubes[1:]:
        cube_lats = cube.coords("latitude")[0].points
        cube_lons = cube.coords("longitude")[0].points
        try:
            np.testing.assert_allclose(ref_lats, cube_lats, rtol=rtol)
            np.testing.assert_allclose(ref_lons, cube_lons, rtol=rtol)
        except AssertionError:
            error_msg = (
                "Cannot unify latitude and longitude, relative difference in "
                "co-ordinates is greater than {}".format(rtol)
            )
            raise ValueError(error_msg)

        lat_dim_numbers = cube.coord_dims("latitude")
        cube.remove_coord("latitude")
        if len(lat_dim_numbers) > 1:
            cube.add_aux_coord(cubes[0].coords("latitude")[0], lat_dim_numbers)
        else:
            cube.add_dim_coord(cubes[0].coords("latitude")[0], lat_dim_numbers[0])

        lon_dim_numbers = cube.coord_dims("longitude")
        cube.remove_coord("longitude")
        if len(lon_dim_numbers) > 1:
            cube.add_aux_coord(cubes[0].coords("longitude")[0], lon_dim_numbers)
        else:
            cube.add_dim_coord(cubes[0].coords("longitude")[0], lon_dim_numbers[0])


def cube_lat_lon_grid_compatible_with_array(cube, array_in):
    """
    Assert that an array can be broadcast onto the cube's lat-lon grid

    Parameters
    ----------
    cube : :obj:`ScmCube`
        :obj:`ScmCube` instance whose lat-lon grid we want to check agains


    array_in : np.ndarray
        The array we want to ensure is able to be broadcast

    Returns
    -------
    bool
        ``True`` if the cube's lat-lon grid is compatible with ``array_in``, otherwise
        ``False``

    Raises
    ------
    AssertionError
        The array cannot be broadcast onto the cube's lat-lon grid
    """
    base_shape = cube.lat_lon_shape

    if array_in.shape != base_shape:
        array_in = np.transpose(array_in)

    if array_in.shape != base_shape:
        return False

    return True


def broadcast_onto_lat_lon_grid(cube, array_in):
    """
    Broadcast an array onto the latitude-longitude grid of ``cube``.

    Here, broadcasting means taking the array and 'duplicating' it so that it
    has the same number of dimensions as the cube's underlying data.

    For example, given a cube with a time dimension of length 3, a latitude dimension of length 4
    and a longitude dimension of length 2 (shape 3x4x2) and ``array_in`` of shape 4x2, results in
    a 3x4x2 array where each slice in the broadcasted array's time dimension is identical to ``array_in``.

    Parameters
    ----------
    cube : :obj:`ScmCube`
        :obj:`ScmCube` instance whose lat-lon grid we want to check agains

    array_in : np.ndarray
        The array we want to broadcast

    Returns
    -------
    array_out
        The original array, broadcast onto the cube's lat-lon grid (i.e. duplicated
        along all dimensions except for latitude and longitude). Note: If the cube has
        lazy data, we return a :obj:`da.Array`, otherwise we return an
        :obj:`np.ndarray`.

    Raises
    ------
    AssertionError
        ``array_in`` cannot be broadcast onto the cube's lat-lon grid because their
        shapes are not compatible

    ValueError
        ``array_in`` cannot be broadcast onto the cube's lat-lon grid by
        ``iris.util.broadcast_to_shape``
    """
    if not cube_lat_lon_grid_compatible_with_array(cube, array_in):
        shape_assert_msg = (
            "the ``array_in`` must be the same shape as the "
            "cube's longitude-latitude grid"
        )
        raise AssertionError(shape_assert_msg)

    broadcaster = np.broadcast_to if not cube.cube.has_lazy_data() else da.broadcast_to
    try:
        return broadcaster(array_in, cube.cube.shape)
    except ValueError as e:
        try_transpose = str(e).startswith(
            "operands could not be broadcast together with remapped shapes"
        ) or str(e).startswith("cannot broadcast shape")
        if not try_transpose:  # pragma: no cover
            raise
        return broadcaster(array_in.T, cube.cube.shape)


def _cftime_conversion(t):
    return dt.datetime(
        t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond
    )


_vector_cftime_conversion = np.vectorize(_cftime_conversion)


def _check_cube_and_adjust_if_needed(cube, time_name="time"):
    """
    Check cube and adjust if required

    Parameters
    ----------
    cube : :obj:`iris.cube.Cube`
        Cube to check

    time_name : str
        Name of the time dimension of the cube to check

    Returns
    -------
    :obj:`iris.cube.Cube`
        Cube, adjusted if needed
    """
    try:
        time_dim = cube.coord(time_name)
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
        logger.warning(warn_msg)
        return _adjust_gregorian_year_zero_units(cube, time_name)

    return cube


def _adjust_gregorian_year_zero_units(cube, time_name):
    """
    Adjust Gregogrian calendar with year zero.

    This function makes the time axis useable with iris again (there is no year zero
    in the Gregorian calendar) at the expense of removing the year zero data.

    Parameters
    ----------
    cube : :obj:`iris.cube.Cube`
        Cube to adjusted

    time_name : str
        Name of the time dimension of the cube to adjust

    Returns
    -------
    :obj:`iris.cube.Cube`
        Adjusted cube

    Raises
    ------
    AssertionError
        Defensive assertion: the code is being used in an unexpected way
    """
    # pylint:disable=too-many-locals
    # hack function to work around very specific use case
    year_zero_cube = cube.copy()
    year_zero_cube_time_dim = cube.coord(time_name)

    gregorian_year_zero_cube = (
        year_zero_cube_time_dim.units.calendar == "gregorian"
    ) and str(year_zero_cube_time_dim.units).startswith("days since 0-1-1")
    if not gregorian_year_zero_cube:  # pragma: no cover # emergency valve
        raise AssertionError("This function is not setup for other cases")

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

    tmp_time_dim = usable_cube.coord(time_name)
    tmp_time = cftime.num2date(
        tmp_time_dim.points, new_unit_str, tmp_time_dim.units.calendar
    )
    # TODO: move to utils
    tmp_time = np.array([dt.datetime(*v.timetuple()[:6]) for v in tmp_time])
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

    new_cube = iris.cube.Cube(usable_data)
    for i, coord in enumerate(usable_cube.dim_coords):
        if coord.standard_name == "time":
            new_cube.add_dim_coord(usable_time_dim, i)
        else:
            new_cube.add_dim_coord(coord, i)

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
        setattr(new_cube, att, getattr(year_zero_cube, att))

    return new_cube
