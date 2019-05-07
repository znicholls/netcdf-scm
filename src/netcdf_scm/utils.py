"""Utils contains a number of helpful functions for doing common cube operations.

For example, applying masks to cubes, taking latitude-longitude means and getting timeseries from a cube as datetime values.
"""
import numpy as np

try:
    import iris
    import cf_units
except ModuleNotFoundError:
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()


def get_cube_timeseries_data(scm_cube):
    """Get a timeseries from a cube.

    This function only works on cubes which are on a time grid only i.e. have no other dimension coordinates.

    Parameters
    ----------
    scm_cube : :obj:`SCMCube`
        An ``SCMCube`` instance with only a 'time' dimension.

    Returns
    -------
    np.ndarray
        The cube's timeseries data.
    """
    _assert_only_cube_dim_coord_is_time(scm_cube)
    return scm_cube.cube.data


def get_scm_cube_time_axis_in_calendar(scm_cube, calendar):
    """Gets a cube's time axis in a given calendar

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
    assert len(scm_cube.cube.dim_coords) == 1, assert_msg
    assert scm_cube.cube.dim_coords[0].standard_name == "time", assert_msg


def assert_all_time_axes_same(time_axes):
    """Assert all time axes in a set are the same.

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
        except AttributeError:  # handle weird numpy error
            raise AssertionError(assert_msg)


def take_lat_lon_mean(in_scmcube, in_weights):
    """Take the latitude longitude mean of a cube with given weights

    Parameters
    ----------
    in_scmcube : :obj:`SCMCube`
        An ``SCMCube`` instance.

    in_weights : np.ndarray
        Weights to use when taking the mean. If you don't have another source, these
        can be generated using
        ``iris.analysis.cartography.area_weights(iris_cube_instance)``

    Returns
    -------
    :obj:`SCMCube`
        A copy of the input cube in which the data is now the latitude-longitude mean
        of the input cube's data
    """
    out_cube = type(in_scmcube)()
    out_cube.cube = in_scmcube.cube.copy()
    out_cube.cube = out_cube.cube.collapsed(
        [in_scmcube.lat_name, in_scmcube.lon_name],
        iris.analysis.MEAN,
        weights=in_weights,
    )
    return out_cube


def apply_mask(in_scmcube, in_mask):
    """Apply a mask to an scm cube's data

    Parameters
    ----------
    in_scmcube : :obj:`SCMCube`
        An ``SCMCube`` instance.

    in_mask : boolean np.ndarray
        The mask to apply

    Returns
    -------
    :obj:`SCMCube`
        A copy of the input cube with the mask applied to its data
    """
    out_cube = type(in_scmcube)()
    out_cube.cube = in_scmcube.cube.copy()
    out_cube.cube.data = np.ma.asarray(out_cube.cube.data)
    out_cube.cube.data.mask = in_mask

    return out_cube


def unify_lat_lon(cubes, rtol=10 ** -10):
    """Unify latitude and longitude co-ordinates of cubes in place.

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
