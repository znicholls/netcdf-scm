import re
from unittest.mock import MagicMock, patch

import iris
import numpy as np
import pytest
from conftest import create_sftlf_cube
from iris.util import broadcast_to_shape

from netcdf_scm.masks import (
    DEFAULT_REGIONS,
    MASKS,
    CubeMasker,
    InvalidMask,
    get_area_mask,
    get_land_mask,
    get_nh_mask,
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
        "World": np.full(nh_mask.shape, False),
        "World|Northern Hemisphere|Land": nh_land_mask,
        "World|Southern Hemisphere|Land": np.logical_or(~nh_mask, land_mask),
        "World|Northern Hemisphere|Ocean": np.logical_or(nh_mask, ~land_mask),
        "World|Southern Hemisphere|Ocean": np.logical_or(~nh_mask, ~land_mask),
        "World|Land": land_mask,
        "World|Ocean": ~land_mask,
        "World|Northern Hemisphere": nh_mask,
        "World|Southern Hemisphere": ~nh_mask,
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


@patch("netcdf_scm.masks.get_nh_mask")
def test_get_scm_masks_no_land_available(mock_nh_mask, test_all_cubes, caplog):
    test_all_cubes.get_metadata_cube = MagicMock(side_effect=OSError)

    nh_mask = np.array(
        [
            [False, False, False, False],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )
    mock_nh_mask.return_value = nh_mask

    expected = {
        "World": np.full(nh_mask.shape, False),
        "World|Northern Hemisphere": nh_mask,
        "World|Southern Hemisphere": ~nh_mask,
    }
    expected_warn = (
        "Land surface fraction (sftlf) data not available, using default instead"
    )
    with patch.dict(MASKS, {"World|Northern Hemisphere": mock_nh_mask}):
        masker = CubeMasker(test_all_cubes)
        result = masker.get_masks(DEFAULT_REGIONS)

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


@pytest.mark.parametrize("inp", ["fail string", np.array([[1, 2], [3, 4]])])
def test_get_land_mask_input_type_errors(test_all_cubes, inp):
    error_msg = re.escape(r"cube must be an SCMCube instance")
    masker = CubeMasker(test_all_cubes)
    with pytest.raises(TypeError, match=error_msg):
        get_land_mask(masker, test_all_cubes, sftlf_cube=inp)


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


def test_get_nh_mask(test_all_cubes):
    result = get_nh_mask(None, test_all_cubes)
    expected_base = np.array(
        [
            [False, False, False, False],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )
    expected = broadcast_to_shape(
        expected_base,
        test_all_cubes.cube.shape,
        [test_all_cubes.lat_dim_number, test_all_cubes.lon_dim_number],
    )

    np.testing.assert_array_equal(result, expected)


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
    # 5N-5S, 170W-120W
    expected_base = np.array(
        [[True, True, True, True], [False, False, False, False], [True, True, True, True]]
    )
    expected = broadcast_to_shape(
        expected_base,
        test_all_cubes.cube.shape,
        [test_all_cubes.lat_dim_number, test_all_cubes.lon_dim_number],
    )

    np.testing.assert_array_equal(result, expected)


def test_area_mask(test_all_cubes):
    # increasing lons (test_nao_mask tests wrapping around)
    result = get_area_mask(-20, 100, 20, 250)(None, test_all_cubes)

    expected_base = np.array(
        [[True, True, True, True], [True, False, False, True], [True, True, True, True]]
    )
    expected = broadcast_to_shape(
        expected_base,
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


def test_unknown_mask_error(test_all_cubes):
    masker = CubeMasker(test_all_cubes)
    with pytest.raises(InvalidMask, match="Unknown mask: junk"):
        masker.get_mask("junk")


def test_get_masks_unknown_mask_warning(test_all_cubes, caplog):
    masker = CubeMasker(test_all_cubes)
    res = masker.get_masks(["World", "junk"])

    assert (~res["World"]).all()

    assert len(caplog.messages) == 1
    assert caplog.messages[0] == "Failed to create junk mask: Unknown mask: junk"
    assert caplog.records[0].levelname == "WARNING"
