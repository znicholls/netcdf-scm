from os.path import join
from unittest.mock import patch, MagicMock
import re
import warnings
import datetime
from dateutil import parser


import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import iris
from iris.util import broadcast_to_shape
import cf_units
import cftime
from openscm.scmdataframe import ScmDataFrame


from netcdf_scm.iris_cube_wrappers import (
    SCMCube,
    MarbleCMIP5Cube,
    CMIP6Input4MIPsCube,
    CMIP6OutputCube,
)
from conftest import (
    TEST_TAS_FILE,
    TEST_SFTLF_FILE,
    TEST_AREACELLA_FILE,
    TEST_ACCESS_CMIP5_FILE,
    tdata_required,
    TEST_DATA_MARBLE_CMIP5_DIR,
    TEST_CMIP6_HISTORICAL_CONCS_FILE,
)


class _SCMCubeIntegrationTester(object):
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

    @tdata_required
    def test_get_scm_timeseries_no_areacealla(self, test_cube):
        var = self.tclass()
        var.cube = iris.load_cube(TEST_TAS_FILE)

        sftlf = self.tclass()
        sftlf.cube = iris.load_cube(TEST_SFTLF_FILE)

        var.get_scm_timeseries(
            sftlf_cube=sftlf, land_mask_threshold=50, areacella_scmcube=None
        )

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
            [test_cube.lat_dim_number, test_cube.lon_dim_number],
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
        if not isinstance(datetimes[0], datetime.datetime):
            datetimes = np.array([parser.parse(x.strftime()) for x in datetimes])
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
                ["unspecified"],
                ["unspecified"],
                ["unspecified"],
            ],
            names=["variable", "unit", "region", "climate_model", "scenario", "model"],
        )
        expected_df = (
            expected_df.unstack().reset_index().rename({0: "value"}, axis="columns")
        )

        expected = ScmDataFrame(expected_df)
        expected.metadata = {"calendar": expected_calendar}

        assert result.metadata == expected.metadata
        assert_frame_equal(result.timeseries(), expected.timeseries())

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


class TestSCMCubeIntegration(_SCMCubeIntegrationTester):
    tclass = SCMCube

    def test_load_and_concatenate_files_in_directory_same_time(self, test_cube):
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

        removed_attributes = ["creation_date", "tracking_id", "history"]
        for removed_attribute in removed_attributes:
            with pytest.raises(KeyError):
                test_cube.cube.attributes[removed_attribute]

    def test_load_and_concatenate_files_in_directory_different_time(self, test_cube):
        tdir = join(
            TEST_DATA_MARBLE_CMIP5_DIR,
            "cmip5",
            "rcp85",
            "Amon",
            "tas",
            "NorESM1-ME",
            "r1i1p1",
        )

        with warnings.catch_warnings(record=True):
            test_cube._load_and_concatenate_files_in_directory(tdir)

        obs_time = test_cube.cube.dim_coords[0]
        obs_time = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )
        assert obs_time[0] == cftime.DatetimeNoLeap(2006, 1, 16, 12, 0, 0, 0, 0, 16)
        assert obs_time[-1] == cftime.DatetimeNoLeap(2100, 12, 16, 12, 0, 0, 0, 1, 350)

        removed_attributes = ["creation_date", "tracking_id", "history"]
        for removed_attribute in removed_attributes:
            with pytest.raises(KeyError):
                test_cube.cube.attributes[removed_attribute]

    def test_load_gregorian_calendar_with_pre_zero_years(self, test_cube):
        expected_warn = (
            "Your calendar is gregorian yet has units of 'days since 0-1-1'. We "
            "rectify this by removing all data before year 1 and changing the units "
            "to 'days since 1-1-1'. If you want other behaviour, you will need to use "
            "another package."
        )
        with warnings.catch_warnings(record=True) as adjust_warnings:
            test_cube.load_data_from_path(TEST_CMIP6_HISTORICAL_CONCS_FILE)

        assert len(adjust_warnings) == 1
        assert str(adjust_warnings[0].message) == expected_warn

        obs_time = test_cube.cube.dim_coords[0]
        assert obs_time.units.name == "day since 1-01-01 00:00:00.000000 UTC"
        assert obs_time.units.calendar == "gregorian"

        obs_time_points = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )
        assert obs_time_points[0] == datetime.datetime(1, 7, 3, 12, 0)
        assert obs_time_points[-1] == datetime.datetime(2014, 7, 3, 12, 0)

        assert test_cube.cube.attributes["institution_id"] == "UoM"
        assert test_cube.cube.attributes["Conventions"] == "CF-1.6"
        assert test_cube.cube.attributes["table_id"] == "input4MIPs"
        assert test_cube.cube.cell_methods[0].method == "mean"
        assert str(test_cube.cube.units) == "1.e-12"
        assert test_cube.cube.var_name == "mole_fraction_of_so2f2_in_air"
        assert test_cube.cube.name() == "mole"
        assert test_cube.cube.long_name == "mole"
        assert isinstance(test_cube.cube.metadata, iris.cube.CubeMetadata)

    def test_access_cmip5_read_issue_30(self, test_cube):
        test_cube.load_data_from_path(TEST_ACCESS_CMIP5_FILE)

        obs_time = test_cube.cube.dim_coords[0]
        assert obs_time.units.name == "day since 1-01-01 00:00:00.000000 UTC"
        assert obs_time.units.calendar == "proleptic_gregorian"

        obs_time_points = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )
        assert obs_time_points[0] == datetime.datetime(2006, 1, 16, 12, 0)
        assert obs_time_points[-1] == datetime.datetime(2049, 12, 16, 12, 0)


class TestMarbleCMIP5Cube(_SCMCubeIntegrationTester):
    tclass = MarbleCMIP5Cube

    def test_load_and_concatenate_files_in_directory_same_time(self, test_cube):
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

        assert test_cube.time_period == "200601-203512"

        removed_attributes = ["creation_date", "tracking_id", "history"]
        for removed_attribute in removed_attributes:
            with pytest.raises(KeyError):
                test_cube.cube.attributes[removed_attribute]

    def test_load_and_concatenate_files_in_directory_different_time(self, test_cube):
        tdir = join(
            TEST_DATA_MARBLE_CMIP5_DIR,
            "cmip5",
            "rcp85",
            "Amon",
            "tas",
            "NorESM1-ME",
            "r1i1p1",
        )

        test_cube._load_and_concatenate_files_in_directory(tdir)

        obs_time = test_cube.cube.dim_coords[0]
        obs_time = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )
        assert obs_time[0] == cftime.DatetimeNoLeap(2006, 1, 16, 12, 0, 0, 0, 0, 16)
        assert obs_time[-1] == cftime.DatetimeNoLeap(2100, 12, 16, 12, 0, 0, 0, 1, 350)

        assert test_cube.time_period == "200601-210012"

        removed_attributes = ["creation_date", "tracking_id", "history"]
        for removed_attribute in removed_attributes:
            with pytest.raises(KeyError):
                test_cube.cube.attributes[removed_attribute]

    def test_get_load_data_from_identifiers_args_from_filepath(self, test_cube):
        tpath = "tests/test_data/marble_cmip5/cmip5/1pctCO2/Amon/fco2antt/CanESM2/r1i1p1/fco2antt_Amon_CanESM2_1pctCO2_r1i1p1_185001-198912.nc"
        expected = {
            "root_dir": "tests/test_data/marble_cmip5",
            "activity": "cmip5",
            "experiment": "1pctCO2",
            "modeling_realm": "Amon",
            "variable_name": "fco2antt",
            "model": "CanESM2",
            "ensemble_member": "r1i1p1",
            "time_period": "185001-198912",
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected

    def test_get_load_data_from_identifiers_args_from_filepath_no_time(self, test_cube):
        tpath = "tests/test_data/marble_cmip5/cmip5/1pctCO2/fx/sftlf/CanESM2/r0i0p0/sftlf_fx_CanESM2_1pctCO2_r0i0p0.nc"
        expected = {
            "root_dir": "tests/test_data/marble_cmip5",
            "activity": "cmip5",
            "experiment": "1pctCO2",
            "modeling_realm": "fx",
            "variable_name": "sftlf",
            "model": "CanESM2",
            "ensemble_member": "r0i0p0",
            "time_period": None,
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected

    def test_get_load_data_from_identifiers_args_from_filepath_no_root_dir(
        self, test_cube
    ):
        tpath = (
            "cmip5/1pctCO2/fx/sftlf/CanESM2/r0i0p0/sftlf_fx_CanESM2_1pctCO2_r0i0p0.nc"
        )
        expected = {
            "root_dir": ".",
            "activity": "cmip5",
            "experiment": "1pctCO2",
            "modeling_realm": "fx",
            "variable_name": "sftlf",
            "model": "CanESM2",
            "ensemble_member": "r0i0p0",
            "time_period": None,
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected

    @pytest.mark.parametrize(
        "tpath",
        ["cmip5/1pctCO2/fx/sftlf/CanESM2/r0i0p0/sftlf_fx_HadGem3_1pctCO2_r0i0p0.nc"],
    )
    def test_get_load_data_from_identifiers_args_from_filepath_errors(
        self, test_cube, tpath
    ):
        error_msg = (
            re.escape("Path and filename do not agree:")
            + "\n"
            + re.escape("    - path model: CanESM2")
            + "\n"
            + re.escape("    - filename model: HadGem3")
            + "\n"
        )
        with pytest.raises(ValueError, match=error_msg):
            test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

    def test_access_cmip5_read_issue_30(self, test_cube):
        test_cube.load_data_from_path(TEST_ACCESS_CMIP5_FILE)

        obs_time = test_cube.cube.dim_coords[0]
        assert obs_time.units.name == "day since 1-01-01 00:00:00.000000 UTC"
        assert obs_time.units.calendar == "proleptic_gregorian"

        obs_time_points = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )
        assert obs_time_points[0] == datetime.datetime(2006, 1, 16, 12, 0)
        assert obs_time_points[-1] == datetime.datetime(2049, 12, 16, 12, 0)

        assert test_cube.model == "ACCESS1-0"


class TestCMIP6Input4MIPsCube(_SCMCubeIntegrationTester):
    tclass = CMIP6Input4MIPsCube

    def test_load_gregorian_calendar_with_pre_zero_years(self, test_cube):
        expected_warn = (
            "Your calendar is gregorian yet has units of 'days since 0-1-1'. We "
            "rectify this by removing all data before year 1 and changing the units "
            "to 'days since 1-1-1'. If you want other behaviour, you will need to use "
            "another package."
        )
        with warnings.catch_warnings(record=True) as adjust_warnings:
            test_cube.load_data_from_path(TEST_CMIP6_HISTORICAL_CONCS_FILE)

        assert len(adjust_warnings) == 1
        assert str(adjust_warnings[0].message) == expected_warn

        obs_time = test_cube.cube.dim_coords[0]
        assert obs_time.units.name == "day since 1-01-01 00:00:00.000000 UTC"
        assert obs_time.units.calendar == "gregorian"

        obs_time_points = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )

        assert obs_time_points[0] == datetime.datetime(1, 7, 3, 12, 0)
        assert obs_time_points[-1] == datetime.datetime(2014, 7, 3, 12, 0)

        assert test_cube.cube.attributes["institution_id"] == "UoM"
        assert test_cube.cube.attributes["Conventions"] == "CF-1.6"
        assert test_cube.cube.attributes["table_id"] == "input4MIPs"
        assert test_cube.cube.cell_methods[0].method == "mean"
        assert str(test_cube.cube.units) == "1.e-12"
        assert test_cube.cube.var_name == "mole_fraction_of_so2f2_in_air"
        assert test_cube.cube.name() == "mole"
        assert test_cube.cube.long_name == "mole"
        assert isinstance(test_cube.cube.metadata, iris.cube.CubeMetadata)

    def test_get_load_data_from_identifiers_args_from_filepath(self, test_cube):
        tpath = "tests/test_data/cmip6-input4mips/input4MIPs/CMIP6/CMIP/PCMDI/PCMDI-AMIP-1-1-4/ocean/mon/tos/gn/v20180427/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc"
        expected = {
            "root_dir": "tests/test_data/cmip6-input4mips",
            "activity_id": "input4MIPs",
            "mip_era": "CMIP6",
            "target_mip": "CMIP",
            "institution_id": "PCMDI",
            "source_id": "PCMDI-AMIP-1-1-4",
            "realm": "ocean",
            "frequency": "mon",
            "variable_id": "tos",
            "grid_label": "gn",
            "version": "v20180427",
            "dataset_category": "SSTsAndSeaIce",
            "time_range": "187001-201712",
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected

    def test_get_load_data_from_identifiers_args_from_filepath_no_time(self, test_cube):
        tpath = "tests/test_data/cmip6-input4mips/input4MIPs/CMIP6/CMIP/PCMDI/PCMDI-AMIP-1-1-4/land/fx/sftlf/gn/v20180427/sftlf_input4MIPs_landState_CMIP_PCMDI-AMIP-1-1-4_gn.nc"
        expected = {
            "root_dir": "tests/test_data/cmip6-input4mips",
            "activity_id": "input4MIPs",
            "mip_era": "CMIP6",
            "target_mip": "CMIP",
            "institution_id": "PCMDI",
            "source_id": "PCMDI-AMIP-1-1-4",
            "realm": "land",
            "frequency": "fx",
            "variable_id": "sftlf",
            "grid_label": "gn",
            "version": "v20180427",
            "dataset_category": "landState",
            "time_range": None,
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected

    def test_get_load_data_from_identifiers_args_from_filepath_no_root_dir(
        self, test_cube
    ):
        tpath = "input4MIPs/CMIP6/CMIP/PCMDI/PCMDI-AMIP-1-1-4/ocean/mon/tos/gn/v20180427/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc"
        expected = {
            "root_dir": ".",
            "activity_id": "input4MIPs",
            "mip_era": "CMIP6",
            "target_mip": "CMIP",
            "institution_id": "PCMDI",
            "source_id": "PCMDI-AMIP-1-1-4",
            "realm": "ocean",
            "frequency": "mon",
            "variable_id": "tos",
            "grid_label": "gn",
            "version": "v20180427",
            "dataset_category": "SSTsAndSeaIce",
            "time_range": "187001-201712",
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected

    @pytest.mark.parametrize(
        "tpath",
        [
            "input4MIPs/CMIP6/CMIP/PCMDI/PCMDI-AMIP-1-1-4/ocean/mon/tas/gn/v20180427/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc"
        ],
    )
    def test_get_load_data_from_identifiers_args_from_filepath_errors(
        self, test_cube, tpath
    ):
        error_msg = (
            re.escape("Path and filename do not agree:")
            + "\n"
            + re.escape("    - path variable_id: tas")
            + "\n"
            + re.escape("    - filename variable_id: tos")
            + "\n"
        )
        with pytest.raises(ValueError, match=error_msg):
            test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)


class TestCMIP6OutputCube(_SCMCubeIntegrationTester):
    tclass = CMIP6OutputCube

    def test_get_load_data_from_identifiers_args_from_filepath(self, test_cube):
        tpath = "tests/test_data/cmip6-output/CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/s1960-r2i1p1f3/day/pr/gn/v20160215/pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f3_gn_198001-198412.nc"
        expected = {
            "root_dir": "tests/test_data/cmip6-output",
            "mip_era": "CMIP6",
            "activity_id": "DCPP",
            "institution_id": "CNRM-CERFACS",
            "source_id": "CNRM-CM6-1",
            "experiment_id": "dcppA-hindcast",
            "member_id": "s1960-r2i1p1f3",
            "table_id": "day",
            "variable_id": "pr",
            "grid_label": "gn",
            "version": "v20160215",
            "time_range": "198001-198412",
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected

    def test_get_load_data_from_identifiers_args_from_filepath_no_time(self, test_cube):
        tpath = "tests/test_data/cmip6-output/CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/r0i0p0/fx/sftlf/gn/v20160215/sftlf_fx_CNRM-CM6-1_dcppA-hindcast_r0i0p0_gn.nc"
        expected = {
            "root_dir": "tests/test_data/cmip6-output",
            "mip_era": "CMIP6",
            "activity_id": "DCPP",
            "institution_id": "CNRM-CERFACS",
            "source_id": "CNRM-CM6-1",
            "experiment_id": "dcppA-hindcast",
            "member_id": "r0i0p0",
            "table_id": "fx",
            "variable_id": "sftlf",
            "grid_label": "gn",
            "version": "v20160215",
            "time_range": None,
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected

    def test_get_load_data_from_identifiers_args_from_filepath_no_root_dir(
        self, test_cube
    ):
        tpath = "CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/s1960-r2i1p1f3/day/pr/gn/v20160215/pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f3_gn_198001-198412.nc"
        expected = {
            "root_dir": ".",
            "mip_era": "CMIP6",
            "activity_id": "DCPP",
            "institution_id": "CNRM-CERFACS",
            "source_id": "CNRM-CM6-1",
            "experiment_id": "dcppA-hindcast",
            "member_id": "s1960-r2i1p1f3",
            "table_id": "day",
            "variable_id": "pr",
            "grid_label": "gn",
            "version": "v20160215",
            "time_range": "198001-198412",
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected

    @pytest.mark.parametrize(
        "tpath",
        [
            "CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/s1960-r2i1p1f3/day/pr/gn/v20160215/pr_day_CNRM-CM6-1_dcppA_s1960-r2i1p1f3_gn_198001-198412.nc"
        ],
    )
    def test_get_load_data_from_identifiers_args_from_filepath_errors(
        self, test_cube, tpath
    ):
        error_msg = (
            re.escape("Path and filename do not agree:")
            + "\n"
            + re.escape("    - path experiment_id: dcppA-hindcast")
            + "\n"
            + re.escape("    - filename experiment_id: dcppA")
            + "\n"
        )
        with pytest.raises(ValueError, match=error_msg):
            test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)
