import re
from unittest.mock import patch

import numpy as np
import pytest
from iris.util import broadcast_to_shape

from netcdf_scm.masks import (
    CubeMasker,
    InvalidMask,
    get_area_mask,
    get_land_mask,
    get_nh_mask,
    or_masks,
    invert,
)


@pytest.mark.parametrize("inp", ["fail string", np.array([[1, 2], [3, 4]])])
def test_get_land_mask_input_type_errors(test_all_cubes, inp):
    error_msg = re.escape(r"cube must be an SCMCube instance")
    masker = CubeMasker(test_all_cubes)
    with pytest.raises(TypeError, match=error_msg):
        get_land_mask(masker, test_all_cubes, sftlf_cube=inp)


def test_get_nh_mask(test_all_cubes):
    masker = CubeMasker(test_all_cubes)
    result = get_nh_mask(masker, test_all_cubes)
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


def test_unknown_mask_error(test_all_cubes):
    masker = CubeMasker(test_all_cubes)
    with pytest.raises(InvalidMask, match="Unknown mask: junk"):
        masker.get_mask("junk")


@patch(
    "netcdf_scm.masks.MASKS",
    {
        "Junk": or_masks(get_area_mask(0, 0, 30, 50), "World|Land"),
        "World|Land": get_land_mask,
        "Inverse": invert("Junk"),
    },
)
def test_no_match_error(test_all_cubes):
    tmask_name = "Junk"

    error_msg = re.escape(
        r"Your cube has no data which matches the `{}` mask".format(tmask_name)
    )
    masker = CubeMasker(test_all_cubes)
    for i in range(3):  # make sure multiple asks still raises
        # should be accessible without issue
        masker.get_mask("World|Land")
        with pytest.raises(ValueError, match=error_msg):
            masker.get_mask("Junk")
        # should be able to get inverse without problem
        res = masker.get_mask("Inverse")
        # inverse of Junk should all be False
        assert not res.any()

