import warnings
from unittest.mock import patch


import pytest
import re
import numpy as np
import datetime
import iris
import cf_units


from netcdf_scm.utils import (
    assert_only_cube_dim_coord_is_time,
    get_cube_timeseries_data,
    get_scm_cube_time_axis_in_calendar,
    assert_all_time_axes_same,
)


def test_assert_only_cube_dim_coord_is_time(test_generic_tas_cube):
    original_cube = test_generic_tas_cube.cube

    error_msg = re.escape("Should only have time coordinate here")

    with pytest.raises(AssertionError, match=error_msg):
        assert_only_cube_dim_coord_is_time(test_generic_tas_cube)

    # can safely ignore these warnings here
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", ".*Collapsing spatial coordinate 'latitude' without weighting.*"
        )
        test_generic_tas_cube.cube = original_cube.collapsed(
            ["latitude", "time"], iris.analysis.MEAN
        )

    with pytest.raises(AssertionError, match=error_msg):
        assert_only_cube_dim_coord_is_time(test_generic_tas_cube)

    # can safely ignore these warnings here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*Using DEFAULT_SPHERICAL.*")
        test_generic_tas_cube.cube = original_cube.collapsed(
            ["latitude", "longitude"],
            iris.analysis.MEAN,
            weights=iris.analysis.cartography.area_weights(original_cube),
        )

    assert_only_cube_dim_coord_is_time(test_generic_tas_cube)


@patch("netcdf_scm.utils.assert_only_cube_dim_coord_is_time")
def test_get_cube_timeseries_data(mock_assert_only_time, test_generic_tas_cube):
    expected = test_generic_tas_cube.cube.data
    result = get_cube_timeseries_data(test_generic_tas_cube)

    np.testing.assert_array_equal(result, expected)
    mock_assert_only_time.assert_called_with(test_generic_tas_cube)


@pytest.mark.parametrize("out_calendar", ["gregorian", "julian", "365_day"])
def test_get_cube_time_axis_in_calendar(test_generic_tas_cube, out_calendar):
    tcn = test_generic_tas_cube.cube.coord_dims("time")[0]
    ttime = test_generic_tas_cube.cube.dim_coords[tcn]
    expected = cf_units.num2date(ttime.points, ttime.units.name, out_calendar)

    result = get_scm_cube_time_axis_in_calendar(test_generic_tas_cube, out_calendar)
    np.testing.assert_array_equal(result, expected)


def test_assert_all_time_axes_same(test_generic_tas_cube):
    tcn = test_generic_tas_cube.cube.coord_dims("time")[0]
    ttime = test_generic_tas_cube.cube.dim_coords[tcn]
    ttime_axis = cf_units.num2date(ttime.points, ttime.units.name, "gregorian")

    assert_all_time_axes_same([ttime_axis, ttime_axis])

    otime_axis = ttime_axis - datetime.timedelta(10)

    error_msg = re.escape("all the time axes should be the same")
    with pytest.raises(AssertionError, match=error_msg):
        assert_all_time_axes_same([otime_axis, ttime_axis])
