from os.path import join
from unittest.mock import patch, MagicMock
import re
import warnings

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import iris
from iris.util import broadcast_to_shape
import cf_units
import cftime
from pymagicc.io import MAGICCData


from netcdf_scm.iris_cube_wrappers import SCMCube, MarbleCMIP5Cube
from conftest import (
    TEST_TAS_FILE,
    TEST_AREACELLA_FILE,
    tdata_required,
    TEST_DATA_MARBLE_CMIP5_DIR,
)


class TestSCMCubeIntegration(object):
    tclass = SCMCube

    @tdata_required
    def test_load_data_from_identifiers_and_areacella(self, test_cube):
        tfile = TEST_TAS_FILE
        test_cube.get_filepath_from_load_data_from_identifiers_args = MagicMock(
            return_value=tfile
        )

        test_constraint = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str("tas"))
        )
        test_cube.get_variable_constraint_from_load_data_from_identifiers_args = MagicMock(
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
            test_cube.load_data_from_identifiers(**tkwargs)

        assert len(record) == 0

        test_cube.get_filepath_from_load_data_from_identifiers_args.assert_called_with(
            **tkwargs
        )
        test_cube.get_variable_constraint_from_load_data_from_identifiers_args.assert_called_with(
            **tkwargs
        )
        test_cube.get_metadata_cube.assert_called_with(test_cube.areacella_var)

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
            "World": np.full(nh_mask.shape, False),
            "World|Northern Hemisphere|Land": np.logical_or(nh_mask, land_mask),
            "World|Southern Hemisphere|Land": np.logical_or(~nh_mask, land_mask),
            "World|Northern Hemisphere|Ocean": np.logical_or(nh_mask, ~land_mask),
            "World|Southern Hemisphere|Ocean": np.logical_or(~nh_mask, ~land_mask),
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

        for label, cube in expected.items():
            assert cube.cube == result[label].cube

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

        test_timeseries_cubes = {
            "World": global_cube,
            "World|Southern Hemisphere|Ocean": sh_ocean_cube,
        }
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("ignore", ".*appropriate model scenario*")
            result = test_cube._convert_scm_timeseries_cubes_to_openscmdata(
                test_timeseries_cubes, out_calendar=out_calendar
            )

        time = sh_ocean_cube.cube.dim_coords[0]
        datetimes = cf_units.num2date(time.points, time.units.name, expected_calendar)
        time_index = pd.Index(datetimes, dtype="object", name="time")

        expected_df = pd.DataFrame(
            {
                "World": global_cube.cube.data,
                "World|Southern Hemisphere|Ocean": sh_ocean_cube.cube.data,
            },
            index=time_index,
        )

        expected_df.columns = pd.MultiIndex.from_product(
            [
                [test_cube.cube.standard_name],
                [test_cube.cube.units.name],
                expected_df.columns.tolist(),
                ["unknown"],
                ["unknown"],
            ],
            names=["variable", "unit", "region", "model", "scenario"],
        )
        expected_df = (
            expected_df.unstack().reset_index().rename({0: "value"}, axis="columns")
        )

        expected = MAGICCData()
        expected.df = expected_df
        expected.metadata = {"calendar": expected_calendar}

        assert result.metadata == expected.metadata
        assert_frame_equal(result.df, expected.df)

    @patch("netcdf_scm.iris_cube_wrappers.os.listdir")
    def test_check_data_names_in_same_directory(self, mock_listdir, test_cube):
        tdir = "mocked"

        mock_listdir.return_value = [
            "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
            "tas_Amon_HadCM3_rcp45_r1i1p1_203101-203512.nc",
            "tas_Amon_HadCM3_rcp45_r1i1p1_203601-203812.nc",
        ]

        test_cube._check_data_names_in_same_directory(tdir)
        mock_listdir.assert_called_with(tdir)

    @patch("netcdf_scm.iris_cube_wrappers.os.listdir")
    @pytest.mark.parametrize(
        "bad_file_list",
        [
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "pr_Amon_HadCM3_rcp45_r1i1p1_203101-203512.nc",
            ],
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "tas_fx_HadCM3_rcp45_r1i1p1_203101-203512.nc",
            ],
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "tas_Amon_CSIRO_rcp45_r1i1p1_203101-203512.nc",
            ],
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "tas_Amon_HadCM3_rcp85_r1i1p1_203101-203512.nc",
            ],
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "tas_Amon_HadCM3_rcp45_r2i1p1_203101-203512.nc",
            ],
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "tas_Amon_HadCM3_rcp45_r1i1p1_203201-203512.nc",
            ],
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "tas_Amon_HadCM3_rcp45_r1i1p1_203012-203512.nc",
            ],
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "pr_Amon_HadCM3_rcp45_r1i1p1_203101-203412.nc",
                "tas_Amon_HadCM3_rcp45_r1i1p1_203601-203812.nc",
            ],
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "tas_Amon_HadCM3_rcp45_r1i1p1_203101-203512.nc",
                "tas_Amon_HadCM3_rcp45_r1i1p1_203701-203812.nc",
            ],
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "tas_Amon_HadCM3_rcp45_r1i1p1_203101-203512.nc",
                "tas_Amon_HadCM3_rcp45_r1i1p1_203602-203812.nc",
            ],
            [
                "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                "tas_Amon_HadCM3_rcp45_r1i1p1_203101-203512.nc",
                "tas_Amon_HadCM3_rcp45_r1i1p2_203601-203812.nc",
            ],
        ],
    )
    def test_check_data_names_in_same_directory_errors(
        self, mock_listdir, bad_file_list, test_cube
    ):
        tdir = "mocked"

        mock_listdir.return_value = bad_file_list
        error_msg = re.escape(
            (
                "Cannot join files in:\n"
                "{}\n"
                "Files found:\n"
                "- {}".format(tdir, "\n- ".join(sorted(bad_file_list)))
            )
        )
        with pytest.raises(AssertionError, match=error_msg):
            test_cube._check_data_names_in_same_directory(tdir)

    def test_load_and_concatenate_files_in_directory(self, test_cube):
        tdir = join(
            TEST_DATA_MARBLE_CMIP5_DIR,
            "cmip5",
            "rcp45",
            "Amon",
            "tas",
            "HadCM3",
            "r1i1p1",
        )

        # can ignore warnings safely here as tested elsewhere
        with warnings.catch_warnings(record=True):
            test_cube._load_and_concatenate_files_in_directory(tdir)

        obs_time = test_cube.cube.dim_coords[0]
        obs_time = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )
        assert obs_time[0] == cftime.Datetime360Day(2006, 1, 16, 0, 0, 0, 0, -1, 16)
        assert obs_time[-1] == cftime.Datetime360Day(2035, 12, 16, 0, 0, 0, 0, -1, 346)

        if type(test_cube) is not SCMCube:
            assert test_cube.time_period == "200601-203512"

        removed_attributes = ["creation_date", "tracking_id", "history"]
        for removed_attribute in removed_attributes:
            with pytest.raises(KeyError):
                test_cube.cube.attributes[removed_attribute]


class TestMarbleCMIP5Cube(TestSCMCubeIntegration):
    tclass = MarbleCMIP5Cube
