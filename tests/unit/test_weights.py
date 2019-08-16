import re
from unittest.mock import patch

import numpy as np
import pytest

from netcdf_scm.weights import (
    CubeWeightCalculator,
    InvalidWeights,
    get_land_weights,
    get_nh_weights,
    get_weights_for_area,
    multiply_weights,
    subtract_weights,
)


@pytest.mark.parametrize("inp", ["fail string", np.array([[1, 2], [3, 4]])])
def test_get_land_mask_input_type_errors(test_all_cubes, inp):
    error_msg = re.escape(r"cube must be an SCMCube instance")
    masker = CubeWeightCalculator(test_all_cubes)
    with pytest.raises(TypeError, match=error_msg):
        get_land_weights(masker, test_all_cubes, sftlf_cube=inp)


def test_get_nh_mask(test_all_cubes):
    masker = CubeWeightCalculator(test_all_cubes)
    result = get_nh_weights(masker, test_all_cubes)
    expected = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])

    np.testing.assert_array_equal(result, expected)


def test_unknown_mask_error(test_all_cubes):
    masker = CubeWeightCalculator(test_all_cubes)
    with pytest.raises(InvalidWeights, match="Unknown weights: junk"):
        masker.get_weights_array("junk")


@patch(
    "netcdf_scm.weights.WEIGHTS_FUNCTIONS_WITHOUT_AREA_WEIGHTING",
    {
        "Junk": multiply_weights(get_weights_for_area(0, 0, 30, 50), "World|Land"),
        "World|Land": get_land_weights,
        "Inverse": subtract_weights("Junk", 1),
    },
)
def test_no_match_error(test_all_cubes):
    tmask_name = "Junk"

    error_msg = re.escape(r"All weights are zero for region: `{}`".format(tmask_name))
    weighter = CubeWeightCalculator(test_all_cubes)
    for i in range(3):  # make sure multiple asks still raises
        # should be accessible without issue
        weighter.get_weights_array("World|Land")
        with pytest.raises(ValueError, match=error_msg):
            weighter.get_weights_array("Junk")
        # should be able to get inverse without problem
        res = weighter.get_weights_array("Inverse")
        # inverse of Junk should all be non-zero
        assert not np.isclose(res, 0).any()
