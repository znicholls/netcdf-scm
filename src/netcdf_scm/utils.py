"""
The utils docstring goes here
"""

import numpy as np
import iris
import cf_units


def get_cube_timeseries_data(scm_cube):
    """

    """
    assert_only_cube_dim_coord_is_time(scm_cube)
    return scm_cube.cube.data


def get_scm_cube_time_axis_in_calendar(scm_cube, calendar):
    """

    """
    time_coord_number = scm_cube.cube.coord_dims("time")[0]
    time = scm_cube.cube.dim_coords[time_coord_number]
    return cf_units.num2date(time.points, time.units.name, calendar)


def assert_only_cube_dim_coord_is_time(scm_cube):
    """

    """
    assert_msg = "Should only have time coordinate here"
    assert len(scm_cube.cube.dim_coords) == 1, assert_msg
    assert scm_cube.cube.dim_coords[0].standard_name == "time", assert_msg


def assert_all_time_axes_same(time_axes):
    """

    """
    for time_axis_to_check in time_axes:
        assert_msg = "all the time axes should be the same"
        np.testing.assert_array_equal(
            time_axis_to_check, time_axes[0], err_msg=assert_msg
        )


def take_lat_lon_mean(in_scmcube, in_weights):
    """

    """
    out_cube = type(in_scmcube)()
    out_cube.cube = in_scmcube.cube.copy()
    out_cube.cube = out_cube.cube.collapsed(
        [in_scmcube._lat_name, in_scmcube._lon_name],
        iris.analysis.MEAN,
        weights=in_weights,
    )
    return out_cube


def apply_mask(in_scmcube, in_mask):
    """move to utils
    """
    out_cube = type(in_scmcube)()
    out_cube.cube = in_scmcube.cube.copy()
    out_cube.cube.data = np.ma.asarray(out_cube.cube.data)
    out_cube.cube.data.mask = in_mask

    return out_cube
