from unittest.mock import MagicMock
import warnings

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import iris
from iris.util import broadcast_to_shape
import cf_units
from pymagicc.io import MAGICCData


from netcdf_scm.iris_cube_wrappers import SCMCube, MarbleCMIP5Cube
from conftest import TEST_TAS_FILE, TEST_AREACELLA_FILE, tdata_required


class TestSCMCubeIntegration(object):
    tclass = SCMCube

    @tdata_required
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

        assert len(record) == 0

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

    @pytest.mark.parametrize("out_calendar", [None, "gregorian", "365_day"])
    def test_convert_scm_timeseries_cubes_to_openscmdata(self, test_cube, out_calendar):
        expected_calendar = (
            test_cube.cube.coords("time")[0].units.calendar
            if out_calendar is None
            else out_calendar
        )

        global_cube = type(test_cube)()
        global_cube.cube = test_cube.cube.copy()
        global_cube.cube.data = 2 * global_cube.cube.data

        # can safely ignore warnings here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*without weighting*")
            global_cube.cube = global_cube.cube.collapsed(
                ["longitude", "latitude"], iris.analysis.MEAN
            )

        sh_ocean_cube = type(test_cube)()
        sh_ocean_cube.cube = test_cube.cube.copy()
        sh_ocean_cube.cube.data = 0.5 * sh_ocean_cube.cube.data
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("ignore", ".*without weighting*")
            sh_ocean_cube.cube = sh_ocean_cube.cube.collapsed(
                ["longitude", "latitude"], iris.analysis.MEAN
            )

        test_timeseries_cubes = {"GLOBAL": global_cube, "SH_OCEAN": sh_ocean_cube}

        result = test_cube._convert_scm_timeseries_cubes_to_openscmdata(
            test_timeseries_cubes, out_calendar=out_calendar
        )

        time = sh_ocean_cube.cube.dim_coords[0]
        datetimes = cf_units.num2date(time.points, time.units.name, expected_calendar)
        time_index = pd.Index(datetimes, dtype="object", name="Time")

        expected_df = pd.DataFrame(
            {"GLOBAL": global_cube.cube.data, "SH_OCEAN": sh_ocean_cube.cube.data},
            index=time_index,
        )

        expected_df.columns = pd.MultiIndex.from_product(
            [
                [test_cube.cube.standard_name],
                [test_cube.cube.units.name],
                expected_df.columns.tolist(),
            ],
            names=["VARIABLE", "UNITS", "REGION"],
        )

        expected = MAGICCData()
        expected.df = expected_df
        expected.metadata = {"calendar": expected_calendar}

        assert result.metadata == expected.metadata
        assert_frame_equal(result.df, expected.df)


class TestMarbleCMIP5Cube(TestSCMCubeIntegration):
    tclass = MarbleCMIP5Cube
