import datetime
import logging
import re
import warnings
from os.path import basename, join
from unittest.mock import MagicMock, patch

import cf_units
import cftime
import iris
import numpy as np
import pandas as pd
import pytest
from conftest import (
    TEST_ACCESS_CMIP5_FILE,
    TEST_AREACELLA_FILE,
    TEST_CMIP6_OUTPUT_FILE,
    TEST_CMIP6_OUTPUT_FILE_1_UNIT,
    TEST_CMIP6_OUTPUT_FILE_HFDS,
    TEST_CMIP6_OUTPUT_FILE_HFDS_NATIVE_GRID,
    TEST_CMIP6_OUTPUT_FILE_THETAO,
    TEST_CMIP6_OUTPUT_FILE_THETAO_NATIVE_GRID,
    TEST_CMIP6_OUTPUT_FILE_MISSING_BOUNDS,
    TEST_CMIP6INPUT4MIPS_HISTORICAL_CONCS_FILE,
    TEST_DATA_MARBLE_CMIP5_DIR,
    TEST_SFTLF_FILE,
    TEST_TAS_FILE,
    assert_scmdata_frames_allclose,
    tdata_required,
)
from dateutil import parser
from iris.exceptions import CoordinateMultiDimError
from iris.util import broadcast_to_shape
from openscm.scmdataframe import ScmDataFrame
from pandas.testing import assert_frame_equal

import netcdf_scm
from netcdf_scm.iris_cube_wrappers import (
    CMIP6Input4MIPsCube,
    CMIP6OutputCube,
    MarbleCMIP5Cube,
    SCMCube,
    _CMIPCube,
)
from netcdf_scm.masks import DEFAULT_REGIONS
from netcdf_scm.utils import broadcast_onto_lat_lon_grid


class _SCMCubeIntegrationTester(object):
    _test_get_scm_timeseries_file = TEST_TAS_FILE

    def test_get_scm_cubes_last_resort(self, test_cube):
        tloaded_paths = ["/path/1", "/path/2"]

        class MockCube:
            @property
            def info(self):
                return {"files": ["/area/cella/a/path/file.nc"]}

        tmetadata_cubes = {"areacella": MockCube()}
        tsftlf_cube = "mocked 124"
        tland_mask_threshold = "mocked 51"

        test_cube._loaded_paths = tloaded_paths
        test_cube._metadata_cubes = tmetadata_cubes

        land_mask = np.array(
            [
                [False, True, True, False],
                [False, True, False, True],
                [False, False, True, False],
            ]
        )
        land_mask = broadcast_onto_lat_lon_grid(test_cube, land_mask)
        nh_mask = np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [True, True, True, True],
            ]
        )
        nh_mask = broadcast_onto_lat_lon_grid(test_cube, nh_mask)

        mocked_masks = {
            "World": np.full(nh_mask.shape, False),
            "World|Land": land_mask,
            "World|Ocean": ~land_mask,
            "World|Northern Hemisphere": nh_mask,
            "World|Southern Hemisphere": ~nh_mask,
            "World|Northern Hemisphere|Land": np.logical_or(nh_mask, land_mask),
            "World|Southern Hemisphere|Land": np.logical_or(~nh_mask, land_mask),
            "World|Northern Hemisphere|Ocean": np.logical_or(nh_mask, ~land_mask),
            "World|Southern Hemisphere|Ocean": np.logical_or(~nh_mask, ~land_mask),
        }
        test_cube._get_scm_masks = MagicMock(return_value=mocked_masks)

        expected = {}
        for label, mask in mocked_masks.items():
            exp_cube = type(test_cube)()

            rcube = test_cube.cube.copy()
            rcube.data.mask = mask
            exp_cube.cube = rcube

            exp_cube.cube.attributes[
                "crunch_netcdf_scm_version"
            ] = "{} (more info at github.com/znicholls/netcdf-scm)".format(
                netcdf_scm.__version__
            )
            exp_cube.cube.attributes[
                "crunch_land_mask_threshold"
            ] = tland_mask_threshold
            exp_cube.cube.attributes[
                "crunch_source_files"
            ] = "Files: {}; areacella: {}".format(
                [basename(p) for p in tloaded_paths],
                [basename(p) for p in tmetadata_cubes["areacella"].info["files"]],
            )
            exp_cube.cube.attributes["variable"] = "air_temperature"
            exp_cube.cube.attributes["variable_standard_name"] = "air_temperature"
            exp_cube.cube.attributes["scenario"] = "unspecified"
            exp_cube.cube.attributes["climate_model"] = "unspecified"
            exp_cube.cube.attributes["member_id"] = "unspecified"
            exp_cube.cube.attributes["mip_era"] = (
                "CMIP5" if isinstance(test_cube, MarbleCMIP5Cube) else "unspecified"
            )
            exp_cube.cube.attributes["activity_id"] = "unspecified"
            exp_cube.cube.attributes["region"] = label
            expected[label] = exp_cube

        result = test_cube.get_scm_cubes(tsftlf_cube, tland_mask_threshold)

        for label, cube in expected.items():
            assert cube.cube.attributes == result[label].cube.attributes
            assert cube.cube == result[label].cube

        test_cube._get_scm_masks.assert_called_with(
            sftlf_cube=tsftlf_cube, land_mask_threshold=tland_mask_threshold, masks=None
        )

    def test_get_scm_timeseries_cubes(self, test_cube):
        tsftlf_cube = "mocked 124"
        tland_mask_threshold = "mocked 51"
        tareacell_scmcube = "mocked 4389"

        land_mask = np.array(
            [
                [False, True, True, False],
                [False, True, False, True],
                [False, False, True, False],
            ]
        )
        land_mask = broadcast_onto_lat_lon_grid(test_cube, land_mask)
        nh_mask = np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [True, True, True, True],
            ]
        )
        nh_mask = broadcast_onto_lat_lon_grid(test_cube, nh_mask)

        mocked_masks = {
            "World": np.full(nh_mask.shape, False),
            "World|Land": land_mask,
            "World|Ocean": ~land_mask,
            "World|Northern Hemisphere": nh_mask,
            "World|Southern Hemisphere": ~nh_mask,
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
        total_area = mocked_weights[0, :, :].sum()
        # do this calculation by hand to be doubly sure, can automate in future if it
        # becomes too annoyting
        ocean_area = 2 + 3 + 4 + 9 + 1
        land_frac = (total_area - ocean_area) / total_area

        land_frac_nh = (
            (~land_mask & ~nh_mask).astype(int) * mocked_weights[0, :, :]
        ).sum() / ((~nh_mask).astype(int) * mocked_weights[0, :, :]).sum()

        land_frac_sh = (
            (~land_mask & nh_mask).astype(int) * mocked_weights[0, :, :]
        ).sum() / ((nh_mask).astype(int) * mocked_weights[0, :, :]).sum()

        test_cube._get_area_weights = MagicMock(return_value=mocked_weights)

        expected = {}
        for label, mask in mocked_masks.items():
            exp_cube = type(test_cube)()

            rcube = test_cube.cube.copy()
            rcube.data.mask = mask
            exp_cube.cube = rcube.collapsed(
                ["latitude", "longitude"], iris.analysis.MEAN, weights=mocked_weights
            )
            exp_cube.cube.add_aux_coord(
                iris.coords.AuxCoord(land_frac, long_name="land_fraction", units=1)
            )
            exp_cube.cube.add_aux_coord(
                iris.coords.AuxCoord(
                    land_frac_nh, long_name="land_fraction_northern_hemisphere", units=1
                )
            )
            exp_cube.cube.add_aux_coord(
                iris.coords.AuxCoord(
                    land_frac_sh, long_name="land_fraction_southern_hemisphere", units=1
                )
            )
            exp_cube.cube.attributes[
                "crunch_land_mask_threshold"
            ] = tland_mask_threshold
            exp_cube.cube.attributes[
                "crunch_netcdf_scm_version"
            ] = "{} (more info at github.com/znicholls/netcdf-scm)".format(
                netcdf_scm.__version__
            )
            exp_cube.cube.attributes["crunch_source_files"] = "Files: []"
            exp_cube.cube.attributes["region"] = label
            exp_cube.cube.attributes.update(test_cube._get_scm_timeseries_ids())
            expected[label] = exp_cube

        result = test_cube.get_scm_timeseries_cubes(
            tsftlf_cube, tland_mask_threshold, tareacell_scmcube
        )

        for label, cube in expected.items():
            assert cube.cube.attributes == result[label].cube.attributes
            assert cube.cube == result[label].cube
            assert result[label].cube.coord("land_fraction").points == land_frac
            assert (
                result[label].cube.coord("land_fraction_northern_hemisphere").points
                == land_frac_nh
            )
            assert (
                result[label].cube.coord("land_fraction_southern_hemisphere").points
                == land_frac_sh
            )

        test_cube._get_scm_masks.assert_called_with(
            sftlf_cube=tsftlf_cube,
            land_mask_threshold=tland_mask_threshold,
            masks=DEFAULT_REGIONS,
        )
        test_cube._get_area_weights.assert_called_with(
            areacell_scmcube=tareacell_scmcube
        )

    @pytest.mark.parametrize("out_calendar", [None, "gregorian", "365_day"])
    def test_convert_scm_timeseries_cubes_to_openscmdata(self, test_cube, out_calendar):
        expected_calendar = (
            test_cube.cube.coords("time")[0].units.calendar
            if out_calendar is None
            else out_calendar
        )

        tmip_era = "tmip"
        test_cube.mip_era = tmip_era

        global_cube = type(test_cube)()
        global_cube.cube = test_cube.cube.copy()
        global_cube.cube.data = 2 * global_cube.cube.data

        # can safely ignore warnings here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*without weighting*")
            global_cube.cube = global_cube.cube.collapsed(
                ["longitude", "latitude"], iris.analysis.MEAN
            )
            global_cube.cube.attributes["region"] = "World"
            global_cube.cube.attributes.update(test_cube._get_scm_timeseries_ids())

        sh_ocean_cube = type(test_cube)()
        sh_ocean_cube.cube = test_cube.cube.copy()
        sh_ocean_cube.cube.data = 0.5 * sh_ocean_cube.cube.data
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("ignore", ".*without weighting*")
            sh_ocean_cube.cube = sh_ocean_cube.cube.collapsed(
                ["longitude", "latitude"], iris.analysis.MEAN
            )
            sh_ocean_cube.cube.attributes["region"] = "World|Southern Hemisphere|Ocean"
            sh_ocean_cube.cube.attributes.update(test_cube._get_scm_timeseries_ids())

        test_timeseries_cubes = {
            "World": global_cube,
            "World|Southern Hemisphere|Ocean": sh_ocean_cube,
        }
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("ignore", ".*appropriate model scenario*")
            result = test_cube.convert_scm_timeseries_cubes_to_openscmdata(
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
                [test_cube.cube.standard_name],
                [str(test_cube.cube.units).replace("-", "^-")],
                expected_df.columns.tolist(),
                ["unspecified"],
                ["unspecified"],
                ["unspecified"],
                ["unspecified"],
                ["unspecified"],
                [tmip_era],
            ],
            names=[
                "variable",
                "variable_standard_name",
                "unit",
                "region",
                "climate_model",
                "scenario",
                "model",
                "activity_id",
                "member_id",
                "mip_era",
            ],
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

    def test_load_gregorian_calendar_with_pre_zero_years(self, test_cube, caplog):
        caplog.set_level(logging.WARNING, logger="netcdf_scm")
        expected_warn = (
            "Your calendar is gregorian yet has units of 'days since 0-1-1'. We "
            "rectify this by removing all data before year 1 and changing the units "
            "to 'days since 1-1-1'. If you want other behaviour, you will need to use "
            "another package."
        )
        test_cube.load_data_from_path(TEST_CMIP6INPUT4MIPS_HISTORICAL_CONCS_FILE)

        # ignore ABCs warning messages
        messages = [m for m in caplog.messages if "ABCs" not in m]
        assert len(messages) == 1
        assert messages[0] == expected_warn
        assert caplog.records[0].levelname == "WARNING"

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


class _CMIPCubeTester(_SCMCubeIntegrationTester):
    tclass = _CMIPCube

    @tdata_required
    def test_load_data_from_identifiers_and_areacella(self, test_cube):
        tfile = TEST_TAS_FILE
        test_cube.get_filepath_from_load_data_from_identifiers_args = MagicMock(
            return_value=tfile
        )

        test_constraint = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str("tas"))
        )
        test_cube.get_variable_constraint = MagicMock(return_value=test_constraint)

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
        test_cube.get_variable_constraint.assert_called_with()
        test_cube.get_metadata_cube.assert_called_with(test_cube.areacella_var)

        cell_measures = test_cube.cube.cell_measures()
        assert len(cell_measures) == 1
        assert cell_measures[0].standard_name == "cell_area"

    @tdata_required
    @pytest.mark.parametrize("force_lazy_load", [True, False])
    def test_get_scm_timeseries(self, test_cube, force_lazy_load):
        var = self.tclass()
        var.load_data_from_path(self._test_get_scm_timeseries_file)

        res = var.get_scm_timeseries()
        assert isinstance(res, ScmDataFrame)

        if force_lazy_load:
            var_lazy = self.tclass()
            var_lazy._crunch_in_memory = MagicMock(side_effect=MemoryError)
            var_lazy.load_data_from_path(self._test_get_scm_timeseries_file)

            res_lazy = var_lazy.get_scm_timeseries()
            assert_scmdata_frames_allclose(res, res_lazy)

    @tdata_required
    def test_get_scm_timeseries_no_areacealla(self, test_cube):
        var = self.tclass()
        var.cube = iris.load_cube(TEST_TAS_FILE)

        sftlf = self.tclass()
        sftlf.cube = iris.load_cube(TEST_SFTLF_FILE)

        var.get_scm_timeseries(
            sftlf_cube=sftlf, land_mask_threshold=50, areacell_scmcube=None
        )

    def test_get_data_reference_syntax(self):
        """
        Test that the cube's data reference syntax is correctly implemented and can be easily accessed

        Should be overwritten in each cube's tester
        """
        assert False


class TestMarbleCMIP5Cube(_CMIPCubeTester):
    tclass = MarbleCMIP5Cube
    _test_get_scm_timeseries_file = TEST_TAS_FILE

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
            "mip_table": "Amon",
            "variable_name": "fco2antt",
            "model": "CanESM2",
            "ensemble_member": "r1i1p1",
            "time_period": "185001-198912",
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected
        assert (
            test_cube.get_filepath_from_load_data_from_identifiers_args(**expected)
            == tpath
        )

    def test_get_load_data_from_identifiers_args_from_filepath_no_time(self, test_cube):
        tpath = "tests/test_data/marble_cmip5/cmip5/1pctCO2/fx/sftlf/CanESM2/r0i0p0/sftlf_fx_CanESM2_1pctCO2_r0i0p0.nc"
        expected = {
            "root_dir": "tests/test_data/marble_cmip5",
            "activity": "cmip5",
            "experiment": "1pctCO2",
            "mip_table": "fx",
            "variable_name": "sftlf",
            "model": "CanESM2",
            "ensemble_member": "r0i0p0",
            "time_period": None,
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected
        assert (
            test_cube.get_filepath_from_load_data_from_identifiers_args(**expected)
            == tpath
        )

    @pytest.mark.parametrize("file_ext", (None, "", ".nc"))
    @pytest.mark.parametrize("time_period", (None, "", "YYYY-YYYY"))
    def test_get_data_reference_syntax(self, file_ext, time_period):
        expected = join(
            "root-dir",
            "activity",
            "experiment",
            "mip-table",
            "variable-name",
            "model",
            "ensemble-member",
            "variable-name_mip-table_model_experiment_ensemble-member_time-periodfile-ext",
        )
        tkwargs = {}
        if file_ext is not None:
            expected = expected.replace("file-ext", file_ext)
            tkwargs["file_ext"] = file_ext
        if time_period is not None:
            expected = expected.replace("time-period", time_period)
        else:
            expected = expected.replace("_time-period", "")
        tkwargs["time_period"] = time_period

        res = self.tclass.get_data_reference_syntax(**tkwargs)
        assert res == expected

    def test_get_load_data_from_identifiers_args_from_filepath_no_root_dir(
        self, test_cube
    ):
        tpath = (
            "./cmip5/1pctCO2/fx/sftlf/CanESM2/r0i0p0/sftlf_fx_CanESM2_1pctCO2_r0i0p0.nc"
        )
        expected = {
            "root_dir": ".",
            "activity": "cmip5",
            "experiment": "1pctCO2",
            "mip_table": "fx",
            "variable_name": "sftlf",
            "model": "CanESM2",
            "ensemble_member": "r0i0p0",
            "time_period": None,
            "file_ext": ".nc",
        }
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

        assert result == expected
        assert (
            test_cube.get_filepath_from_load_data_from_identifiers_args(**expected)
            == tpath
        )

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


class TestCMIP6Input4MIPsCube(_CMIPCubeTester):
    tclass = CMIP6Input4MIPsCube
    _test_get_scm_timeseries_file = None  # I don't have any test files for this

    def test_load_gregorian_calendar_with_pre_zero_years(self, test_cube, caplog):
        caplog.set_level(logging.WARNING, logger="netcdf_scm")
        expected_warn = (
            "Your calendar is gregorian yet has units of 'days since 0-1-1'. We "
            "rectify this by removing all data before year 1 and changing the units "
            "to 'days since 1-1-1'. If you want other behaviour, you will need to use "
            "another package."
        )
        test_cube.load_data_from_path(TEST_CMIP6INPUT4MIPS_HISTORICAL_CONCS_FILE)

        # ignore ABCs warning messages
        messages = [m for m in caplog.messages if "ABCs" not in m]
        assert len(messages) == 1
        assert messages[0] == expected_warn
        assert caplog.records[0].levelname == "WARNING"

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

    @pytest.mark.parametrize("file_ext", (None, "", ".nc"))
    @pytest.mark.parametrize("time_period", (None, "", "YYYY-YYYY"))
    def test_get_data_reference_syntax(self, file_ext, time_period):
        expected = join(
            "root-dir",
            "activity-id",
            "mip-era",
            "target-mip",
            "institution-id",
            "source-id",
            "realm",
            "frequency",
            "variable-id",
            "grid-label",
            "version",
            "variable-id_activity-id_dataset-category_target-mip_source-id_grid-label_time-rangefile-ext",
        )
        tkwargs = {}
        if file_ext is not None:
            expected = expected.replace("file-ext", file_ext)
            tkwargs["file_ext"] = file_ext
        if time_period is not None:
            expected = expected.replace("time-range", time_period)
        else:
            expected = expected.replace("_time-range", "")
        tkwargs["time_range"] = time_period

        res = self.tclass.get_data_reference_syntax(**tkwargs)
        assert res == expected

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
        assert (
            test_cube.get_filepath_from_load_data_from_identifiers_args(**expected)
            == tpath
        )

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
        assert (
            test_cube.get_filepath_from_load_data_from_identifiers_args(**expected)
            == tpath
        )

    def test_get_load_data_from_identifiers_args_from_filepath_no_root_dir(
        self, test_cube
    ):
        tpath = "./input4MIPs/CMIP6/CMIP/PCMDI/PCMDI-AMIP-1-1-4/ocean/mon/tos/gn/v20180427/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc"
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
        assert (
            test_cube.get_filepath_from_load_data_from_identifiers_args(**expected)
            == tpath
        )

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

    def test_get_scm_timeseries(self):
        pytest.skip("No test data included at the moment")


class TestCMIP6OutputCube(_CMIPCubeTester):
    tclass = CMIP6OutputCube
    _test_get_scm_timeseries_file = TEST_CMIP6_OUTPUT_FILE

    @pytest.mark.parametrize("file_ext", (None, "", ".nc"))
    @pytest.mark.parametrize("time_period", (None, "", "YYYY-YYYY"))
    def test_get_data_reference_syntax(self, file_ext, time_period):
        expected = join(
            "root-dir",
            "mip-era",
            "activity-id",
            "institution-id",
            "source-id",
            "experiment-id",
            "member-id",
            "table-id",
            "variable-id",
            "grid-label",
            "version",
            "variable-id_table-id_source-id_experiment-id_member-id_grid-label_time-rangefile-ext",
        )
        tkwargs = {}
        if file_ext is not None:
            expected = expected.replace("file-ext", file_ext)
            tkwargs["file_ext"] = file_ext
        if time_period is not None:
            expected = expected.replace("time-range", time_period)
        else:
            expected = expected.replace("_time-range", "")
        tkwargs["time_range"] = time_period

        res = self.tclass.get_data_reference_syntax(**tkwargs)
        assert res == expected

    def test_get_load_data_from_identifiers_args_from_filepath(self, test_cube):
        tpath = "./tests/test_data/cmip6-output/CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/s1960-r2i1p1f3/day/pr/gn/v20160215/pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f3_gn_198001-198412.nc"
        expected = {
            "root_dir": "./tests/test_data/cmip6-output",
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
        assert (
            test_cube.get_filepath_from_load_data_from_identifiers_args(**expected)
            == tpath
        )

    def test_get_load_data_from_identifiers_args_from_filepath_no_time(self, test_cube):
        tpath = "./tests/test_data/cmip6-output/CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/r0i0p0/fx/sftlf/gn/v20160215/sftlf_fx_CNRM-CM6-1_dcppA-hindcast_r0i0p0_gn.nc"
        expected = {
            "root_dir": "./tests/test_data/cmip6-output",
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
        assert (
            test_cube.get_filepath_from_load_data_from_identifiers_args(**expected)
            == tpath
        )

    def test_get_load_data_from_identifiers_args_from_filepath_no_root_dir(
        self, test_cube
    ):
        tpath = "./CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/s1960-r2i1p1f3/day/pr/gn/v20160215/pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f3_gn_198001-198412.nc"
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
        assert (
            test_cube.get_filepath_from_load_data_from_identifiers_args(**expected)
            == tpath
        )

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

    def test_load_data(self, test_cube):
        test_cube.load_data_from_path(TEST_CMIP6_OUTPUT_FILE)

        obs_time = test_cube.cube.dim_coords[0]
        assert obs_time.units.name == "day since 1850-01-01 00:00:00.0000000 UTC"
        assert obs_time.units.calendar == "365_day"

        obs_time_points = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )

        assert obs_time_points[0] == cftime.DatetimeNoLeap(
            1850, 1, 16, 12, 0, 0, 0, 5, 16
        )
        assert obs_time_points[-1] == cftime.DatetimeNoLeap(
            1859, 12, 16, 12, 0, 0, 0, 5, 350
        )

        assert test_cube.cube.attributes["institution_id"] == "BCC"
        assert test_cube.cube.attributes["Conventions"] == "CF-1.5"
        assert test_cube.cube.attributes["table_id"] == "Amon"
        assert test_cube.cube.cell_methods[0].method == "mean"
        assert str(test_cube.cube.units) == "W m-2"
        assert test_cube.cube.var_name == "rlut"
        assert test_cube.cube.name() == "toa_outgoing_longwave_flux"
        assert test_cube.cube.long_name == "TOA Outgoing Longwave Radiation"
        assert isinstance(test_cube.cube.metadata, iris.cube.CubeMetadata)

        ts = test_cube.get_scm_timeseries()
        assert (ts["model"] == "unspecified").all()
        assert (ts["scenario"] == "1pctCO2").all()
        assert (ts["activity_id"] == "CMIP").all()
        assert (ts["member_id"] == "r1i1p1f1").all()
        assert (
            ts["region"]
            == [
                "World",
                "World|Land",
                "World|Ocean",
                "World|Northern Hemisphere",
                "World|Southern Hemisphere",
                "World|Northern Hemisphere|Land",
                "World|Southern Hemisphere|Land",
                "World|Northern Hemisphere|Ocean",
                "World|Southern Hemisphere|Ocean",
            ]
        ).all()
        assert (ts["variable"] == "rlut").all()
        assert (ts["variable_standard_name"] == "toa_outgoing_longwave_flux").all()
        assert (ts["unit"] == "W m^-2").all()
        assert (ts["climate_model"] == "BCC-CSM2-MR").all()

    def test_load_data_missing_bounds(self, test_cube):
        test_cube.load_data_from_path(TEST_CMIP6_OUTPUT_FILE_MISSING_BOUNDS)

        obs_time = test_cube.cube.dim_coords[0]
        assert obs_time.units.name == "day since 2015-01-01 00:00:00.00000000 UTC"
        assert obs_time.units.calendar == "gregorian"

        obs_time_points = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )

        assert obs_time_points[0] == datetime.datetime(2015, 1, 16, 12, 0)
        assert obs_time_points[-1] == datetime.datetime(2100, 12, 16, 12, 0)

        assert test_cube.cube.attributes["institution_id"] == "IPSL"
        assert test_cube.cube.attributes["Conventions"] == "CF-1.7 CMIP-6.2"
        assert test_cube.cube.attributes["table_id"] == "Lmon"
        assert test_cube.cube.cell_methods[0].method == "mean where land"
        assert str(test_cube.cube.units) == "kg m-2"
        assert test_cube.cube.var_name == "cSoilFast"
        assert test_cube.cube.name() == "fast_soil_pool_mass_content_of_carbon"
        assert test_cube.cube.long_name == "Carbon Mass in Fast Soil Pool"
        assert isinstance(test_cube.cube.metadata, iris.cube.CubeMetadata)

        ts = test_cube.get_scm_timeseries()
        assert (ts["model"] == "unspecified").all()
        assert (ts["scenario"] == "ssp126").all()
        assert (ts["activity_id"] == "ScenarioMIP").all()
        assert (ts["member_id"] == "r1i1p1f1").all()
        assert sorted(ts["region"].tolist()) == sorted(
            [
                "World",
                "World|Land",
                "World|Northern Hemisphere",
                "World|Northern Hemisphere|Land",
                "World|Northern Hemisphere|Ocean",
                "World|Ocean",
                "World|Southern Hemisphere",
                "World|Southern Hemisphere|Land",
                "World|Southern Hemisphere|Ocean",
            ]
        )
        assert (ts["variable"] == "cSoilFast").all()
        assert (
            ts["variable_standard_name"] == "fast_soil_pool_mass_content_of_carbon"
        ).all()
        assert (ts["unit"] == "kg m^-2").all()
        assert (ts["climate_model"] == "IPSL-CM6A-LR").all()

    @pytest.mark.parametrize(
        "tdata_path",
        [TEST_CMIP6_OUTPUT_FILE_HFDS, TEST_CMIP6_OUTPUT_FILE_HFDS_NATIVE_GRID],
    )
    def test_load_hfds_data(self, test_cube, tdata_path):
        test_cube.load_data_from_path(tdata_path)

        obs_time = test_cube.cube.dim_coords[0]
        assert obs_time.units.name == "day since 1-01-01 00:00:00.000000 UTC"
        assert obs_time.units.calendar == "365_day"

        obs_time_points = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )

        assert obs_time_points[0] == cftime.DatetimeNoLeap(
            1850, 1, 15, 13, 0, 0, 0, -1, 1
        )
        assert obs_time_points[-1] == cftime.DatetimeNoLeap(
            1850, 6, 15, 0, 0, 0, 0, 1, 166
        )

        assert isinstance(test_cube.cube.metadata, iris.cube.CubeMetadata)

        error_msg = re.escape(
            "Your cube has no data which matches the `World|Land` mask"
        )
        with pytest.raises(ValueError, match=error_msg):
            test_cube.get_scm_timeseries(masks=["World|Land"])

        ts = test_cube.get_scm_timeseries(
            masks=[
                "World",
                "World|Northern Hemisphere",
                "World|Northern Hemisphere|Ocean",
                "World|Ocean",
                "World|Southern Hemisphere",
                "World|Southern Hemisphere|Ocean",
                "World|North Atlantic Ocean",
                "World|El Nino N3.4",
            ]
        )
        assert sorted(ts["region"].tolist()) == sorted(
            [
                "World",
                "World|Northern Hemisphere",
                "World|Northern Hemisphere|Ocean",
                "World|Ocean",
                "World|Southern Hemisphere",
                "World|Southern Hemisphere|Ocean",
                "World|North Atlantic Ocean",
                "World|El Nino N3.4",
            ]
        )
        assert (ts["variable"] == "hfds").all()
        assert (
            ts["variable_standard_name"] == "surface_downward_heat_flux_in_sea_water"
        ).all()
        assert (ts["unit"] == "W m^-2").all()
        assert (ts["climate_model"] == "CESM2").all()
        np.testing.assert_allclose(
            ts.filter(region="World|El Nino N3.4", month=3).values.squeeze(),
            131.46116737,
            rtol=0.01,
        )

    @patch.object(tclass, "_get_areacell_scmcube", return_value=None)
    def test_load_hfds_data_native_grid_no_areacello_error(
        self, mock_get_areacell_scmcube, test_cube
    ):
        test_cube.load_data_from_path(TEST_CMIP6_OUTPUT_FILE_HFDS_NATIVE_GRID)
        error_msg = re.escape(
            "iris does not yet support multi-dimensional co-ordinates, you will "
            "need your data's cell area information before you can crunch"
        )
        with pytest.raises(CoordinateMultiDimError, match=error_msg):
            test_cube.get_scm_timeseries(
                masks=[
                    "World",
                    "World|Northern Hemisphere",
                    "World|Northern Hemisphere|Ocean",
                ]
            )

    @pytest.mark.parametrize(
        "tdata_path",
        [TEST_CMIP6_OUTPUT_FILE_THETAO, TEST_CMIP6_OUTPUT_FILE_THETAO_NATIVE_GRID],
    )
    def test_load_thetao_data(self, test_cube, tdata_path):
        test_cube.load_data_from_path(tdata_path)

        obs_time = test_cube.cube.dim_coords[0]
        assert obs_time.units.name == "day since 1-01-01 00:00:00.000000 UTC"
        assert obs_time.units.calendar == "365_day"

        obs_time_points = cf_units.num2date(
            obs_time.points, obs_time.units.name, obs_time.units.calendar
        )

        assert obs_time_points[0] == cftime.DatetimeNoLeap(1, 1, 15, 13, 0, 0, 0, -1, 1)
        assert obs_time_points[-1] == cftime.DatetimeNoLeap(1, 3, 15, 12, 0, 0, 0, 6, 74)

        assert isinstance(test_cube.cube.metadata, iris.cube.CubeMetadata)

        error_msg = re.escape(
            "Your cube has no data which matches the `World|Land` mask"
        )
        with pytest.raises(ValueError, match=error_msg):
            test_cube.get_scm_timeseries(masks=["World|Land"])

        ts = test_cube.get_scm_timeseries(
            masks=[
                "World",
                "World|Northern Hemisphere",
                "World|Northern Hemisphere|Ocean",
                "World|Ocean",
                "World|Southern Hemisphere",
                "World|Southern Hemisphere|Ocean",
                "World|North Atlantic Ocean",
                "World|El Nino N3.4",
            ]
        )
        assert sorted(ts["region"].tolist()) == sorted(
            [
                "World",
                "World|Northern Hemisphere",
                "World|Northern Hemisphere|Ocean",
                "World|Ocean",
                "World|Southern Hemisphere",
                "World|Southern Hemisphere|Ocean",
                "World|North Atlantic Ocean",
                "World|El Nino N3.4",
            ]
        )
        assert (ts["variable"] == "hfds").all()
        assert (
            ts["variable_standard_name"] == "surface_downward_heat_flux_in_sea_water"
        ).all()
        assert (ts["unit"] == "W m^-2").all()
        assert (ts["climate_model"] == "CESM2").all()
        np.testing.assert_allclose(
            ts.filter(region="World|El Nino N3.4", month=3).values.squeeze(),
            131.46116737,
            rtol=0.01,
        )

    @tdata_required
    def test_load_data_auto_add_areacella(self, test_cube):
        test_cube.load_data_from_path(TEST_CMIP6_OUTPUT_FILE_HFDS)

        cell_measures = test_cube.cube.cell_measures()
        assert len(cell_measures) == 1
        assert cell_measures[0].standard_name == "cell_area"

    @tdata_required
    def test_load_data_auto_add_areacello_and_volcello(self, test_cube):
        test_cube.load_data_from_path(TEST_CMIP6_OUTPUT_FILE_THETAO_NATIVE_GRID)

        cell_measures = test_cube.cube.cell_measures()
        assert len(cell_measures) == 2
        assert cell_measures[0].standard_name == "cell_area"
        assert cell_measures[0].standard_name == "cell_volume"

    def test_load_data_1_unit(self, test_cube):
        test_cube.load_data_from_path(TEST_CMIP6_OUTPUT_FILE_1_UNIT)

        ts = test_cube.get_scm_timeseries()
        assert (ts["unit"] == "dimensionless").all()
        assert (ts["climate_model"] == "CNRM-CM6-1").all()
