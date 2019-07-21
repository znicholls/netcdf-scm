import logging
import re
from unittest.mock import MagicMock, patch

import iris
import numpy as np
import pytest
from conftest import create_sftlf_cube
from iris.util import broadcast_to_shape

from netcdf_scm.iris_cube_wrappers import SCMCube
from netcdf_scm.masks import (
    DEFAULT_REGIONS,
    MASKS,
    CubeMasker,
    get_area_mask,
    get_default_sftlf_cube,
    get_land_mask,
)


@patch("netcdf_scm.masks.get_land_mask")
@patch("netcdf_scm.masks.get_nh_mask")
def test_get_scm_masks(mock_nh_mask, mock_land_mask, test_all_cubes):
    tsftlf_cube = "mocked 124"
    tland_mask_threshold = "mocked 51"

    land_mask = np.array(
        [
            [False, True, True, False],
            [False, True, False, True],
            [False, False, True, False],
        ]
    )
    mock_land_mask.return_value = land_mask

    nh_mask = np.array(
        [
            [False, False, False, False],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )
    mock_nh_mask.return_value = nh_mask

    nh_land_mask = np.array(
        [
            [False, True, True, False],
            [False, True, False, True],
            [True, True, True, True],
        ]
    )
    # check our logic while we're here
    np.testing.assert_array_equal(np.logical_or(nh_mask, land_mask), nh_land_mask)

    expected = {
        k: broadcast_to_shape(
            v,
            test_all_cubes.cube.shape,
            [test_all_cubes.lat_dim_number, test_all_cubes.lon_dim_number],
        )
        for k, v in {
            "World": np.full(nh_mask.shape, False),
            "World|Northern Hemisphere|Land": nh_land_mask,
            "World|Southern Hemisphere|Land": np.logical_or(~nh_mask, land_mask),
            "World|Northern Hemisphere|Ocean": np.logical_or(nh_mask, ~land_mask),
            "World|Southern Hemisphere|Ocean": np.logical_or(~nh_mask, ~land_mask),
            "World|Land": land_mask,
            "World|Ocean": ~land_mask,
            "World|Northern Hemisphere": nh_mask,
            "World|Southern Hemisphere": ~nh_mask,
        }.items()
    }

    with patch.dict(
        MASKS, {"World|Northern Hemisphere": mock_nh_mask, "World|Land": mock_land_mask}
    ):
        masker = CubeMasker(
            test_all_cubes,
            sftlf_cube=tsftlf_cube,
            land_mask_threshold=tland_mask_threshold,
        )
        result = masker.get_masks(DEFAULT_REGIONS)

    for label, array in expected.items():
        np.testing.assert_array_equal(array, result[label])
    mock_land_mask.assert_called_with(
        masker,
        test_all_cubes,
        sftlf_cube=tsftlf_cube,
        land_mask_threshold=tland_mask_threshold,
    )
    mock_nh_mask.assert_called_with(
        masker,
        test_all_cubes,
        sftlf_cube=tsftlf_cube,
        land_mask_threshold=tland_mask_threshold,
    )


@pytest.mark.parametrize("with_bounds", [True, False])
@patch("netcdf_scm.masks.get_nh_mask")
def test_get_scm_masks_no_land_available(
    mock_nh_mask, with_bounds, test_all_cubes, caplog
):
    test_all_cubes.get_metadata_cube = MagicMock(side_effect=OSError)

    nh_mask = np.array(
        [
            [False, False, False, False],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )
    mock_nh_mask.return_value = nh_mask
    default_sftlf_cube = get_default_sftlf_cube()
    default_sftlf_cube = default_sftlf_cube.regrid(
        # AreaWeighted() in future but too slow now
        test_all_cubes.cube,
        iris.analysis.Linear(),
    )
    expected_land_mask = ~(default_sftlf_cube.data > 50).data

    expected = {
        k: broadcast_to_shape(
            v,
            test_all_cubes.cube.shape,
            [test_all_cubes.lat_dim_number, test_all_cubes.lon_dim_number],
        )
        for k, v in {
            "World": np.full(nh_mask.shape, False),
            "World|Northern Hemisphere": nh_mask,
            "World|Northern Hemisphere|Land": ~(~nh_mask & ~expected_land_mask),
            "World|Southern Hemisphere": ~nh_mask,
        }.items()
    }
    expected_warn = (
        "Land surface fraction (sftlf) data not available, using default instead"
    )
    with patch.dict(MASKS, {"World|Northern Hemisphere": mock_nh_mask}):
        if not with_bounds:
            test_all_cubes.lat_dim.bounds = None
            test_all_cubes.lon_dim.bounds = None
        masker = CubeMasker(test_all_cubes)
        result = masker.get_masks(expected.keys())

    assert len(caplog.messages) == 1
    assert caplog.messages[0] == expected_warn

    for label, array in expected.items():
        np.testing.assert_array_equal(array, result[label])
    mock_nh_mask.assert_called_with(masker, test_all_cubes)


@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("input_format", ["scmcube", None])
@pytest.mark.parametrize("sftlf_var", ["sftlf", "sftlf_other"])
@pytest.mark.parametrize(
    "test_threshold",
    [(None), (0), (10), (30), (49), (49.9), (50), (50.1), (51), (60), (75), (100)],
)
def test_get_land_mask(
    test_all_cubes, test_threshold, input_format, sftlf_var, transpose
):
    sftlf_cube = create_sftlf_cube(test_all_cubes.__class__)
    test_all_cubes.sftlf_var = sftlf_var
    original_data = sftlf_cube.cube.data

    if transpose:
        sftlf_cube.cube = iris.cube.Cube(data=np.transpose(sftlf_cube.cube.data))
    test_all_cubes.get_metadata_cube = MagicMock(return_value=sftlf_cube)

    test_land_fraction_input = sftlf_cube if input_format == "scmcube" else None

    masker = CubeMasker(test_all_cubes)
    if test_threshold is None:
        result = get_land_mask(
            masker, test_all_cubes, sftlf_cube=test_land_fraction_input
        )
        # test that default land fraction is 50%
        test_threshold = 50
    else:
        result = get_land_mask(
            masker,
            test_all_cubes,
            sftlf_cube=test_land_fraction_input,
            land_mask_threshold=test_threshold,
        )

    # where it's land return False, otherwise True to match with masking
    # convention that True means masked
    expected = broadcast_to_shape(
        np.where(original_data > test_threshold, False, True),
        test_all_cubes.cube.shape,
        [test_all_cubes.lat_dim_number, test_all_cubes.lon_dim_number],
    )
    np.testing.assert_array_equal(result, expected)
    # Check that the sftlf meta cube is always registered
    test_all_cubes.get_metadata_cube.assert_called_with(
        sftlf_var, cube=test_land_fraction_input
    )


def test_get_land_mask_shape_errors(test_all_cubes):
    sftlf_cube = create_sftlf_cube(test_all_cubes.__class__)
    error_msg = re.escape(
        r"the sftlf_cube data must be the same shape as the "
        r"cube's longitude-latitude grid"
    )

    wrong_shape_data = np.array([[1, 2], [3, 4]])
    sftlf_cube.cube = iris.cube.Cube(data=wrong_shape_data)
    masker = CubeMasker(test_all_cubes)
    with pytest.raises(AssertionError, match=error_msg):
        get_land_mask(masker, test_all_cubes, sftlf_cube=sftlf_cube)

    test_all_cubes.get_metadata_cube = MagicMock(return_value=sftlf_cube)
    with pytest.raises(AssertionError, match=error_msg):
        get_land_mask(masker, test_all_cubes, sftlf_cube=None)


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


def test_nao_mask(test_all_cubes):
    sftlf_cube = create_sftlf_cube(test_all_cubes.__class__)
    masker = CubeMasker(test_all_cubes, sftlf_cube=sftlf_cube, land_mask_threshold=50.5)
    result = masker.get_mask("World|North Atlantic Ocean")

    expected_base = np.array(
        [[True, True, True, False], [True, True, True, True], [True, True, True, True]]
    )
    expected = broadcast_to_shape(
        expected_base,
        test_all_cubes.cube.shape,
        [test_all_cubes.lat_dim_number, test_all_cubes.lon_dim_number],
    )

    np.testing.assert_array_equal(result, expected)


def test_elnino_mask(test_all_cubes):
    sftlf_cube = create_sftlf_cube(test_all_cubes.__class__)
    masker = CubeMasker(test_all_cubes, sftlf_cube=sftlf_cube, land_mask_threshold=50.5)
    result = masker.get_mask("World|El Nino N3.4")
    # 5N-5S, 190E-240E
    expected_base = np.array(
        [[True, True, True, True], [True, True, False, True], [True, True, True, True]]
    )
    expected = broadcast_to_shape(
        expected_base,
        test_all_cubes.cube.shape,
        [test_all_cubes.lat_dim_number, test_all_cubes.lon_dim_number],
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
            np.array(
                [
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, True],
                ]
            ),
        ),
        (  # edge of bound included negative co-ord
            np.array([66, 0, -60]),
            np.array([-135, -80, 45, 135]),
            np.array(
                [
                    [True, True, True, True],
                    [True, False, True, True],
                    [True, True, True, True],
                ]
            ),
        ),
        (  # one within bounds
            np.array([80, 35, -70]),
            np.array([10, 30, 50, 135, 320]),
            np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, True, False],
                    [True, True, True, True, True],
                ]
            ),
        ),
        (  # one within bounds negative co-ord
            np.array([80, 35, -70]),
            np.array([-95, -40, 40, 135]),
            np.array(
                [
                    [True, True, True, True],
                    [True, False, True, True],
                    [True, True, True, True],
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("query", [[0, -80, 65, 0], [0, 280, 65, 360]])
def test_area_mask(test_all_cubes, query, lat_pts, lon_pts, expected):
    regrid_cube = create_dummy_cube_from_lat_lon_points(lat_pts, lon_pts)
    test_all_cubes.cube = test_all_cubes.cube.regrid(
        regrid_cube, iris.analysis.Linear()
    )

    if isinstance(expected, str) and expected == "error":
        error_msg = re.compile("None of the cube's.*lie within the bounds.*")
        with pytest.raises(ValueError, match=error_msg):
            get_area_mask(*query)(None, test_all_cubes)
        return

    result = get_area_mask(*query)(None, test_all_cubes)

    expected = broadcast_to_shape(
        expected,
        test_all_cubes.cube.shape,
        [test_all_cubes.lat_dim_number, test_all_cubes.lon_dim_number],
    )
    np.testing.assert_array_equal(result, expected)


def test_area_mask_wrapped_lons(test_all_cubes):
    result = get_area_mask(0, -80, 65, 0)(None, test_all_cubes)

    expected_base = np.array(
        [[True, True, True, False], [True, True, True, False], [True, True, True, True]]
    )
    expected = broadcast_to_shape(
        expected_base,
        test_all_cubes.cube.shape,
        [test_all_cubes.lat_dim_number, test_all_cubes.lon_dim_number],
    )

    np.testing.assert_array_equal(result, expected)


def test_get_masks_unknown_mask_warning(test_all_cubes, caplog):
    masker = CubeMasker(test_all_cubes)
    res = masker.get_masks(["World", "junk"])

    assert (~res["World"]).all()

    assert len(caplog.messages) == 1
    assert caplog.messages[0] == "Failed to create junk mask: Unknown mask: junk"
    assert caplog.records[0].levelname == "WARNING"


@pytest.mark.parametrize(
    "exp_warn,cube_max,land_mask_threshold",
    [(False, 100, 50), (True, 100, 0.5), (False, 1, 0.5), (True, 1, 50)],
)
def test_get_scm_masks_land_bound_checks(
    exp_warn, cube_max, land_mask_threshold, test_all_cubes, caplog
):
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
    masker = CubeMasker(test_all_cubes, land_mask_threshold=land_mask_threshold)
    masker.get_masks(["World|Land"])

    if exp_warn:
        assumed_land_mask_threshold = (
            land_mask_threshold / 100
            if land_mask_threshold > 1
            else land_mask_threshold * 100
        )
        expected_warn = "sftlf data max is {} and requested land_mask_threshold is {}, assuming land_mask_threshold should be {}".format(
            tsftlf_cube.data.max(), land_mask_threshold, assumed_land_mask_threshold
        )
        assert len(caplog.messages) == 1
        assert expected_warn == caplog.messages[0]
    else:
        assert len(caplog.messages) == 0
