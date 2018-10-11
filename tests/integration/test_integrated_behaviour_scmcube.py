from unittest.mock import MagicMock

import pytest
import numpy as np
import iris
from iris.util import broadcast_to_shape


from netcdf_scm.iris_cube_wrappers import SCMCube, MarbleCMIP5Cube
from conftest import TEST_TAS_FILE, TEST_AREACELLA_FILE


class TestSCMCubeIntegration(object):
    tclass = SCMCube

    def test_load_data_and_areacella(self, test_cube):
        tfile = TEST_TAS_FILE
        test_cube.get_file_from_load_data_args = MagicMock(return_value=tfile)

        test_constraint = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str("tas"))
        )
        test_cube.get_variable_constraint_from_load_data_args = MagicMock(
            return_value=test_constraint
        )

        tmdata_scmcube = type(test_cube)()
        tmdata_scmcube.cube = iris.load_cube(TEST_AREACELLA_FILE)
        test_cube.get_metadata_cube = MagicMock(return_value=tmdata_scmcube)

        tkwargs = {
            "variable_name": "fco2antt",
            "modeling_realm": "Amon",
            "model": "CanESM2",
            "experiment": "1pctCO2",
        }

        with pytest.warns(None) as record:
            test_cube.load_data(**tkwargs)
        # Gracefully filling warnings, when we move to iris v2.2.0, change this to zero
        # as that bug will be fixed
        assert len(record) == 6

        test_cube.get_file_from_load_data_args.assert_called_with(**tkwargs)
        test_cube.get_variable_constraint_from_load_data_args.assert_called_with(
            **tkwargs
        )
        test_cube.get_metadata_cube.assert_called_with(test_cube._areacella_var)

        cell_measures = test_cube.cube.cell_measures()
        assert len(cell_measures) == 1
        assert cell_measures[0].standard_name == "cell_area"

    def test_get_scm_timeseries_cubes(self, test_cube):
        tsftlf_cube = "mocked 124"
        tland_mask_threshold = "mocked 51"
        tareacella_scmcube = "mocked 4389"

        land_mask = np.array(
            [
                [False, True, True, False],
                [False, True, False, True],
                [False, False, True, False],
            ]
        )
        nh_mask = np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [True, True, True, True],
            ]
        )

        mocked_masks = {
            "GLOBAL": np.full(nh_mask.shape, False),
            "NH_LAND": np.logical_or(nh_mask, land_mask),
            "SH_LAND": np.logical_or(~nh_mask, land_mask),
            "NH_OCEAN": np.logical_or(nh_mask, ~land_mask),
            "SH_OCEAN": np.logical_or(~nh_mask, ~land_mask),
        }
        test_cube._get_scm_masks = MagicMock(return_value=mocked_masks)

        mocked_weights = broadcast_to_shape(
            np.array([[1, 2, 3, 4], [1, 4, 8, 9], [0, 4, 1, 9]]),
            test_cube.cube.shape,
            [test_cube._lat_dim_number, test_cube._lon_dim_number],
        )

        test_cube._get_area_weights = MagicMock(return_value=mocked_weights)

        expected = {}
        for label, mask in mocked_masks.items():
            exp_cube = type(test_cube)()

            rcube = test_cube.cube.copy()
            rcube.data.mask = mask
            exp_cube.cube = rcube.collapsed(
                ["latitude", "longitude"], iris.analysis.MEAN, weights=mocked_weights
            )
            expected[label] = exp_cube

        result = test_cube.get_scm_timeseries_cubes(
            tsftlf_cube, tland_mask_threshold, tareacella_scmcube
        )

        for label, cube in result.items():
            assert cube.cube == expected[label].cube

        test_cube._get_scm_masks.assert_called_with(
            sftlf_cube=tsftlf_cube, land_mask_threshold=tland_mask_threshold
        )
        test_cube._get_area_weights.assert_called_with(
            areacella_scmcube=tareacella_scmcube
        )


class TestMarbleCMIP5Cube(TestSCMCubeIntegration):
    tclass = MarbleCMIP5Cube
