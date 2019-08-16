"""
Utils contains a number of helpful functions for doing common cube operations.

For example, applying masks to cubes, taking latitude-longitude means and getting
timeseries from a cube as datetime values.
"""
import datetime as dt

import numpy as np
import numpy.ma as ma

try:
    import dask.array as da
    import iris
    from iris.analysis import WeightedAggregator, _build_dask_mdtol_function
    from iris.util import broadcast_to_shape
    import cf_units

    # monkey patch iris MEAN until https://github.com/SciTools/iris/pull/3299 is merged
    iris.analysis.MEAN = WeightedAggregator(
        "mean", ma.average, lazy_func=_build_dask_mdtol_function(da.ma.average)
    )

except ModuleNotFoundError:  # pragma: no cover # emergency valve
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()


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
    time_coord_number = scm_cube.cube.coord_dims("time")[0]
    time = scm_cube.cube.dim_coords[time_coord_number]
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

        lat_dim_no = cube.coord_dims("latitude")[0]
        cube.remove_coord("latitude")
        cube.add_dim_coord(cubes[0].coords("latitude")[0], lat_dim_no)

        lon_dim_no = cube.coord_dims("longitude")[0]
        cube.remove_coord("longitude")
        cube.add_dim_coord(cubes[0].coords("longitude")[0], lon_dim_no)


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
    lat_length = len(cube.lat_dim.points)
    lon_length = len(cube.lon_dim.points)

    base_shape = (lat_length, lon_length)
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
    np.ndarray
        The original array, broadcast onto the cube's lat-lon grid (i.e. duplicated
        along all dimensions except for latitude and longitude)

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

    dim_order = [cube.lat_dim_number, cube.lon_dim_number]
    try:
        return broadcast_to_shape(array_in, cube.cube.shape, dim_order)
    except ValueError as e:
        if str(e) != "shape and array are not compatible":  # pragma: no cover
            raise
        return broadcast_to_shape(array_in.T, cube.cube.shape, dim_order)


def _cftime_conversion(t):
    return dt.datetime(
        t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond
    )


_vector_cftime_conversion = np.vectorize(_cftime_conversion)
