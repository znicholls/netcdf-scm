import logging
import re
from unittest.mock import MagicMock, PropertyMock, patch

import iris
import numpy as np
import pytest
from conftest import create_sftlf_cube

from netcdf_scm.iris_cube_wrappers import SCMCube
from netcdf_scm.weights import (
    DEFAULT_REGIONS,
    WEIGHTS_FUNCTIONS_WITHOUT_AREA_WEIGHTING,
    CubeWeightCalculator,
    get_default_sftlf_cube,
    get_land_weights,
    get_weights_for_area,
)


@patch("netcdf_scm.weights.get_land_weights")
@patch("netcdf_scm.weights.get_nh_weights")
def test_get_scm_masks(mock_nh_weights, mock_land_weights, test_all_cubes):
    tsftlf_cube = "mocked 124"

    land_weights = np.array([[1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 1]])
    mock_land_weights.return_value = land_weights

    nh_weights = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    mock_nh_weights.return_value = nh_weights

    nh_land_weights = np.array([[1, 0, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0]])
    # check our logic while we're here
    np.testing.assert_array_equal(nh_weights * land_weights, nh_land_weights)

    area_weights = test_all_cubes.get_area_weights()
    expected = {
        k: area_weights * v
        for k, v in {
            "World": np.full(nh_weights.shape, 1),
            "World|Northern Hemisphere|Land": nh_land_weights,
            "World|Southern Hemisphere|Land": (1 - nh_weights) * land_weights,
            "World|Northern Hemisphere|Ocean": nh_weights * (100 - land_weights),
            "World|Southern Hemisphere|Ocean": (1 - nh_weights) * (100 - land_weights),
            "World|Land": land_weights,
            "World|Ocean": 100 - land_weights,
            "World|Northern Hemisphere": nh_weights,
            "World|Southern Hemisphere": 1 - nh_weights,
        }.items()
    }

    with patch.dict(
        WEIGHTS_FUNCTIONS_WITHOUT_AREA_WEIGHTING,
        {"World|Northern Hemisphere": mock_nh_weights, "World|Land": mock_land_weights},
    ):
        masker = CubeWeightCalculator(test_all_cubes, sftlf_cube=tsftlf_cube)
        result = masker.get_weights(DEFAULT_REGIONS)

    for label, array in expected.items():
        np.testing.assert_array_equal(array, result[label])
    mock_land_weights.assert_called_with(masker, test_all_cubes, sftlf_cube=tsftlf_cube)
    mock_nh_weights.assert_called_with(masker, test_all_cubes, sftlf_cube=tsftlf_cube)


@pytest.mark.parametrize("with_bounds", [True, False])
@patch("netcdf_scm.weights.get_nh_weights")
def test_get_scm_masks_no_land_available(
    mock_nh_weights, with_bounds, test_all_cubes, caplog
):
    caplog.set_level(logging.WARNING, logger="netcdf_scm.iris_cube_wrappers")
    test_all_cubes.get_metadata_cube = MagicMock(side_effect=OSError)

    nh_weights = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    mock_nh_weights.return_value = nh_weights
    default_sftlf_cube = get_default_sftlf_cube()
    default_sftlf_cube = default_sftlf_cube.regrid(
        # AreaWeighted() in future but too slow now
        test_all_cubes.cube,
        iris.analysis.Linear(),
    )
    expected_land_weights = default_sftlf_cube.data

    area_weights = test_all_cubes.get_area_weights()
    expected = {
        k: area_weights * v
        for k, v in {
            "World": np.full(nh_weights.shape, 1),
            "World|Northern Hemisphere": nh_weights,
            "World|Northern Hemisphere|Land": nh_weights * expected_land_weights,
            "World|Southern Hemisphere": 1 - nh_weights,
        }.items()
    }
    expected_warn = (
        "Land surface fraction (sftlf) data not available, using default instead"
    )
    with patch.dict(
        WEIGHTS_FUNCTIONS_WITHOUT_AREA_WEIGHTING,
        {"World|Northern Hemisphere": mock_nh_weights},
    ):
        if not with_bounds:
            test_all_cubes.lat_dim.bounds = None
            test_all_cubes.lon_dim.bounds = None
        masker = CubeWeightCalculator(test_all_cubes)
        result = masker.get_weights(expected.keys())

    if not with_bounds:
        assert len(caplog.messages) == 4
        assert (
            caplog.messages[0]
            == "Couldn't find/use areacell_cube, falling back to iris.analysis.cartography.area_weights"
        )
        assert caplog.messages[0] == caplog.messages[1]
        assert caplog.messages[2] == "Guessing latitude and longitude bounds"
        assert caplog.messages[3] == expected_warn
    else:
        assert len(caplog.messages) == 3
        assert (
            caplog.messages[0]
            == "Couldn't find/use areacell_cube, falling back to iris.analysis.cartography.area_weights"
        )
        assert caplog.messages[0] == caplog.messages[1]
        assert caplog.messages[2] == expected_warn

    for label, array in expected.items():
        np.testing.assert_array_equal(array, result[label])
    mock_nh_weights.assert_called_with(masker, test_all_cubes)


@pytest.mark.parametrize("input_format", ["scmcube", None])
@pytest.mark.parametrize("sftlf_var", ["sftlf", "sftlf_other"])
@patch(
    "netcdf_scm.iris_cube_wrappers.SCMCube.surface_fraction_var",
    new_callable=PropertyMock,
)
def test_get_land_weights(
    mock_surface_fraction_var, test_all_cubes, input_format, sftlf_var
):
    sftlf_cube = create_sftlf_cube(test_all_cubes.__class__)
    mock_surface_fraction_var.return_value = sftlf_var
    original_data = sftlf_cube.cube.data

    test_all_cubes.get_metadata_cube = MagicMock(return_value=sftlf_cube)

    test_land_fraction_input = sftlf_cube if input_format == "scmcube" else None

    masker = CubeWeightCalculator(test_all_cubes)
    result = get_land_weights(
        masker, test_all_cubes, sftlf_cube=test_land_fraction_input
    )

    expected = original_data
    np.testing.assert_array_equal(result, expected)
    # Check that the sftlf meta cube is always registered
    test_all_cubes.get_metadata_cube.assert_called_with(
        sftlf_var, cube=test_land_fraction_input
    )


def test_get_land_weights_shape_errors(test_all_cubes):
    sftlf_cube = create_sftlf_cube(test_all_cubes.__class__)
    error_msg = re.escape(
        r"the sftlf_cube data must be the same shape as the "
        r"cube's longitude-latitude grid"
    )

    wrong_shape_data = np.array([[1, 2], [3, 4]])
    sftlf_cube.cube = iris.cube.Cube(data=wrong_shape_data)
    masker = CubeWeightCalculator(test_all_cubes)
    with pytest.raises(AssertionError, match=error_msg):
        get_land_weights(masker, test_all_cubes, sftlf_cube=sftlf_cube)

    test_all_cubes.get_metadata_cube = MagicMock(return_value=sftlf_cube)
    with pytest.raises(AssertionError, match=error_msg):
        get_land_weights(masker, test_all_cubes, sftlf_cube=None)


def create_dummy_cube_from_lat_lon_points(lat_pts, lon_pts):
    lat = iris.coords.DimCoord(lat_pts, standard_name="latitude", units="degrees")
    lon = iris.coords.DimCoord(
        lon_pts, standard_name="longitude", units="degrees", circular=True
    )

    cube = iris.cube.Cube(
        np.full((len(lat_pts), len(lon_pts)), 0),
        dim_coords_and_dims=[(lat, 0), (lon, 1)],
    )

    return cube


def test_nao_weights(test_all_cubes):
    sftlf_cube = create_sftlf_cube(test_all_cubes.__class__)
    masker = CubeWeightCalculator(test_all_cubes, sftlf_cube=sftlf_cube)
    result = masker.get_weights_array_without_area_weighting(
        "World|North Atlantic Ocean"
    )

    expected = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]) * (
        100 - sftlf_cube.cube.data
    )

    np.testing.assert_array_equal(result, expected)


def test_elnino_weights(test_all_cubes):
    sftlf_cube = create_sftlf_cube(test_all_cubes.__class__)
    masker = CubeWeightCalculator(test_all_cubes, sftlf_cube=sftlf_cube)
    result = masker.get_weights_array_without_area_weighting("World|El Nino N3.4")
    # 5N-5S, 190E-240E
    expected = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]) * (
        100 - sftlf_cube.cube.data
    )

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "lat_pts,lon_pts,expected",
    [
        (  # nothing within bounds, raises Error
            np.array([-60, -1, 80]),
            np.array([45, 135, 225, 315]),
            "error",
        ),
        (  # nothing within bounds, raises Error
            np.array([-60, 10, 80]),
            np.array([45, 135, 225, 279]),
            "error",
        ),
        (  # nothing within bounds negative co-ord, raises Error
            np.array([-60, -1, 80]),
            np.array([-135, -45, 45, 135]),
            "error",
        ),
        (  # edge of bound included
            np.array([65, 0, -60]),
            np.array([45, 135, 225, 280]),
            np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]),
        ),
        (  # edge of bound included negative co-ord
            np.array([66, 0, -60]),
            np.array([-135, -80, 45, 135]),
            np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
        ),
        (  # one within bounds
            np.array([80, 35, -70]),
            np.array([10, 30, 50, 135, 320]),
            np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]),
        ),
        (  # one within bounds negative co-ord
            np.array([80, 35, -70]),
            np.array([-95, -40, 40, 135]),
            np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
        ),
    ],
)
@pytest.mark.parametrize("query", [[0, -80, 65, 0], [0, 280, 65, 360]])
def test_get_weights_for_area(test_all_cubes, query, lat_pts, lon_pts, expected):
    regrid_cube = create_dummy_cube_from_lat_lon_points(lat_pts, lon_pts)
    test_all_cubes.cube = test_all_cubes.cube.regrid(
        regrid_cube, iris.analysis.Linear()
    )

    if isinstance(expected, str) and expected == "error":
        error_msg = re.compile("None of the cube's.*lie within the bounds.*")
        with pytest.raises(ValueError, match=error_msg):
            get_weights_for_area(*query)(None, test_all_cubes)
        return

    result = get_weights_for_area(*query)(None, test_all_cubes)
    np.testing.assert_array_equal(result, expected)


def test_area_mask_wrapped_lons(test_all_cubes):
    result = get_weights_for_area(0, -80, 65, 0)(None, test_all_cubes)

    expected = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])

    np.testing.assert_array_equal(result, expected)


def test_get_masks_unknown_weights_warning(test_all_cubes, caplog):
    caplog.set_level(logging.WARNING, logger="netcdf_scm.iris_cube_wrappers")
    masker = CubeWeightCalculator(test_all_cubes)
    res = masker.get_weights(["World", "junk"])

    np.testing.assert_allclose(res["World"], test_all_cubes.get_area_weights())

    assert len(caplog.messages) == 3
    assert (
        caplog.messages[0]
        == "Couldn't find/use areacell_cube, falling back to iris.analysis.cartography.area_weights"
    )
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.messages[1] == "Failed to create junk weights: Unknown weights: junk"
    assert caplog.records[1].levelname == "WARNING"
    assert caplog.messages[0] == caplog.messages[2]
    assert caplog.records[0].levelname == caplog.records[2].levelname


@pytest.mark.parametrize("exp_warn,cube_max", [(False, 100), (True, 1)])
def test_get_scm_weights_land_bound_checks(exp_warn, cube_max, test_all_cubes, caplog):
    tsftlf_cube = get_default_sftlf_cube().regrid(
        test_all_cubes.cube, iris.analysis.Linear()
    )
    tsftlf_cube_max = tsftlf_cube.data.max()
    assert np.isclose(tsftlf_cube_max, 100)
    if cube_max == 1:
        tsftlf_cube.data = tsftlf_cube.data / 100

    test_all_cubes.get_metadata_cube = MagicMock()
    tsftlf_scmcube = SCMCube()
    tsftlf_scmcube.cube = tsftlf_cube
    test_all_cubes.get_metadata_cube.return_value = tsftlf_scmcube

    caplog.set_level(logging.INFO)
    masker = CubeWeightCalculator(test_all_cubes)
    masker.get_weights(["World|Land"])

    if exp_warn:
        expected_warn = "sftlf data max is {}, multiplying by 100 to convert units to percent".format(
            tsftlf_cube.data.max()
        )
        assert len(caplog.messages) == 2
        assert caplog.messages[1] == expected_warn
    else:
        assert len(caplog.messages) == 1

    assert (
        caplog.messages[0]
        == "No `realm` attribute in `self.cube`, guessing the data is in the realm `atmos`"
    )
