import datetime as dt
import logging
import re
import warnings
from os.path import basename, dirname, join
from unittest.mock import MagicMock, PropertyMock, call, patch

import cftime
import iris
import numpy as np
import pandas as pd
import pytest
from iris.exceptions import ConstraintMismatchError
from pandas.testing import assert_frame_equal, assert_index_equal

from netcdf_scm.iris_cube_wrappers import (
    CMIP6Input4MIPsCube,
    CMIP6OutputCube,
    MarbleCMIP5Cube,
    SCMCube,
    _CMIPCube,
)
from netcdf_scm.weights import DEFAULT_REGIONS, CubeWeightCalculator


class TestSCMCube(object):
    tclass = SCMCube

    attributes_to_set_from_fixtures = {}

    @pytest.fixture(autouse=True)
    def auto_injector_fixture(self, request):
        data = self.attributes_to_set_from_fixtures
        for attribute_to_set, fixture_name in data.items():
            setattr(self, attribute_to_set, request.getfixturevalue(fixture_name))

    def run_test_of_method_to_overload(
        self, test_cube, method_to_overload, junk_args={}
    ):
        if type(test_cube) is SCMCube:
            with pytest.raises(NotImplementedError):
                getattr(test_cube, method_to_overload)(**junk_args)
        else:
            self.raise_overload_message(method_to_overload)

    def raise_overload_message(self, method_to_overload):
        assert False, "Overload {} in your subclass".format(method_to_overload)

    @patch("netcdf_scm.iris_cube_wrappers.iris.load_cube")
    @patch("netcdf_scm.iris_cube_wrappers._check_cube_and_adjust_if_needed")
    @pytest.mark.parametrize(
        "tfilepath,tconstraint",
        [("here/there/now.nc", "mocked"), ("here/there/now.nc", None)],
    )
    def test_load_cube(
        self,
        mock_check_cube_and_adjust_if_needed,
        mock_iris_load_cube,
        test_cube,
        tfilepath,
        tconstraint,
    ):
        test_cube._check_cube = MagicMock()
        if tconstraint is not None:
            test_cube._load_cube(tfilepath, constraint=tconstraint)
        else:
            test_cube._load_cube(tfilepath)
        mock_iris_load_cube.assert_called_with(tfilepath, constraint=tconstraint)
        mock_check_cube_and_adjust_if_needed.assert_called()

    @patch.object(SCMCube, "get_metadata_cube")
    def test_process_load_data_from_identifiers_warnings(
        self, mock_get_metadata_cube, test_cube, caplog
    ):
        mock_get_metadata_cube.side_effect = ValueError("mocked error")

        warn_1 = "warning 1"
        warn_2 = "warning 2"
        warn_area = (
            "PATH-STAMP Missing CF-netCDF measure variable 'areacella' other stuff"
        )
        with warnings.catch_warnings(record=True) as mock_warn_no_area:
            warnings.warn(warn_1)
            warnings.warn(warn_2)

        caplog.clear()
        test_cube._process_load_data_from_identifiers_warnings(mock_warn_no_area)

        assert len(caplog.messages) == 2  # just rethrow warnings
        assert caplog.messages[0] == warn_1
        assert caplog.messages[1] == warn_2
        for record in caplog.records:
            assert record.levelname == "WARNING"

        with warnings.catch_warnings(record=True) as mock_warn_area:
            warnings.warn(warn_1)
            warnings.warn(warn_2)
            warnings.warn(warn_area)

        caplog.clear()
        caplog.set_level(logging.DEBUG)
        test_cube._process_load_data_from_identifiers_warnings(mock_warn_area)

        # warnings plus extra exception and guess of realm
        assert len(caplog.messages) == 4
        assert caplog.messages[0] == warn_1
        assert caplog.records[0].levelname == "WARNING"
        assert caplog.messages[1] == warn_2
        assert caplog.records[1].levelname == "WARNING"
        assert caplog.records[2].levelname == "INFO"
        assert "NetCDF-SCM will treat the data as `atmosphere`" in str(
            caplog.records[2].message
        )
        assert caplog.records[3].levelname == "DEBUG"
        assert warn_area in str(caplog.records[3].message)

    def test_add_areacell_measure(self, test_cube, test_areacella_file, test_tas_file):
        # can safely ignore warnings here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*Missing CF-netCDF measure.*")
            test_cube.cube = iris.load_cube(test_tas_file)

        tareacellacube = type(test_cube)()

        tareacellacube.cube = iris.load_cube(test_areacella_file)
        test_cube.get_metadata_cube = MagicMock(return_value=tareacellacube)

        test_cube._add_areacell_measure("not used", test_cube.areacell_var)

        assert any(["area" in cm.measure for cm in test_cube.cube.cell_measures()])
        assert any(
            ["cell_area" in cm.standard_name for cm in test_cube.cube.cell_measures()]
        )

    def test_load_data_from_path(self, test_cube):
        if type(test_cube) is SCMCube:
            test_cube._load_cube = MagicMock()
            tpath = "here/there/everywehre/test.nc"
            test_cube.load_data_from_path(tpath)
            test_cube._load_cube.assert_called_with(tpath)
        else:
            self.raise_overload_message("test_load_data")

    @pytest.mark.parametrize("process_warnings", [True, False])
    def test_load_data_in_directory(self, test_cube, process_warnings):
        tdir = "mocked/out"
        test_cube._load_and_concatenate_files_in_directory = MagicMock()
        test_cube.load_data_in_directory(tdir, process_warnings=process_warnings)
        test_cube._load_and_concatenate_files_in_directory.assert_called_with(
            tdir, process_warnings=process_warnings
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("regions", [None, ["World"]])
    def test_get_scm_timeseries(self, test_sftlf_cube, test_cube, regions, lazy):
        tsftlf_cube = "mocked 124"
        tareacell_scmcube = "mocked 4389"

        exp_regions = DEFAULT_REGIONS if regions is None else regions
        test_cubes_return = {m: 3 for m in exp_regions}
        test_cube.get_scm_timeseries_cubes = MagicMock(return_value=test_cubes_return)

        test_conversion_return = pd.DataFrame(data=np.array([1, 2, 3]))
        test_cube.convert_scm_timeseries_cubes_to_openscmdata = MagicMock(
            return_value=test_conversion_return
        )

        result = test_cube.get_scm_timeseries(
            surface_fraction_cube=tsftlf_cube,
            areacell_scmcube=tareacell_scmcube,
            regions=regions,
            lazy=lazy,
        )

        test_cube.get_scm_timeseries_cubes.assert_called_with(
            surface_fraction_cube=tsftlf_cube,
            areacell_scmcube=tareacell_scmcube,
            regions=exp_regions,
            lazy=lazy,
        )
        test_cube.convert_scm_timeseries_cubes_to_openscmdata.assert_called_with(
            test_cubes_return
        )

        assert_frame_equal(result, test_conversion_return)

    def test_get_scm_timeseries_ids_warnings(
        self, test_cube, caplog, expected_mip_era="unspecified", expected_warns=6
    ):
        warn_msg = re.compile("Could not determine .*, filling with 'unspecified'")
        warn_msg_variable = "Could not determine variable, filling with standard_name"
        res = test_cube._get_scm_timeseries_ids()

        assert res["climate_model"] == "unspecified"
        assert res["scenario"] == "unspecified"
        assert res["activity_id"] == "unspecified"
        assert res["member_id"] == "unspecified"
        assert res["mip_era"] == expected_mip_era
        assert res["variable"] == "air_temperature"
        assert res["variable_standard_name"] == "air_temperature"

        assert len(caplog.messages) == expected_warns
        for m in caplog.messages:
            assert warn_msg.match(str(m)) or str(m) == warn_msg_variable

    def test_get_scm_timeseries_ids(self, test_cube, caplog):
        tmodel = "ABCD"
        tactivity = "rcpmip"
        tensemble_member = "r1i3p10"
        tscenario = "oscvolcanicrf"
        tmip_era = "CMIP3"
        test_cube.model = tmodel
        test_cube.activity = tactivity
        test_cube.experiment = tscenario
        test_cube.ensemble_member = tensemble_member
        test_cube.mip_era = tmip_era

        res = test_cube._get_scm_timeseries_ids()
        assert res["climate_model"] == tmodel
        assert res["scenario"] == tscenario
        assert res["activity_id"] == tactivity
        assert res["member_id"] == tensemble_member
        assert res["mip_era"] == tmip_era

    @pytest.mark.parametrize("tregions", (None, ["a", "b", "custom", "World|Land"]))
    @pytest.mark.parametrize("tareacell_scmcube", (None, "mocked areacell cube"))
    @pytest.mark.parametrize("tsftlf_scmcube", (None, "mocked sftlf cube"))
    @patch.object(CubeWeightCalculator, "get_weights")
    @patch.object(CubeWeightCalculator, "__init__")
    @patch.object(SCMCube, "get_metadata_cube")
    def test_get_scm_timeseries_weights(
        self,
        mock_get_metadata_cube,
        mock_weight_calculator_init,
        mock_get_weights,
        test_cube,
        tsftlf_scmcube,
        tareacell_scmcube,
        tregions,
    ):
        tgetweights_return = "mock return"
        mock_get_weights.return_value = tgetweights_return
        mock_weight_calculator_init.return_value = None

        res = test_cube.get_scm_timeseries_weights(
            surface_fraction_cube=tsftlf_scmcube,
            areacell_scmcube=tareacell_scmcube,
            regions=tregions,
        )

        assert res == tgetweights_return

        if tareacell_scmcube is not None:
            mock_get_metadata_cube.assert_has_calls(
                [call(test_cube.areacell_var, cube=tareacell_scmcube)]
            )

        if tsftlf_scmcube is not None:
            mock_get_metadata_cube.assert_has_calls(
                [call(test_cube.surface_fraction_var, cube=tsftlf_scmcube)]
            )

        expected_regions = tregions if tregions is not None else DEFAULT_REGIONS
        mock_get_weights.assert_called_with(expected_regions)
        assert mock_weight_calculator_init.call_count == 1
        mock_weight_calculator_init.assert_called_with(test_cube)
        # test calling again does not call masker again
        test_cube.get_scm_timeseries_weights(
            surface_fraction_cube=tsftlf_scmcube,
            areacell_scmcube=tareacell_scmcube,
            regions=tregions,
        )
        assert mock_weight_calculator_init.call_count == 1

    @pytest.mark.parametrize("input_format", ["scmcube", None])
    @pytest.mark.parametrize("areacell_var", ["areacella", "area_other"])
    @patch(
        "netcdf_scm.iris_cube_wrappers.SCMCube.areacell_var", new_callable=PropertyMock
    )
    def test_get_area_weights(
        self, mock_areacell_var, test_cube, test_sftlf_cube, areacell_var, input_format
    ):
        mock_areacell_var.return_value = areacell_var

        test_sftlf_cube.cube.units = "m**2"

        expected = test_sftlf_cube.cube.data

        test_cube._get_areacell_scmcube = MagicMock(return_value=test_sftlf_cube)

        test_areacella_input = test_sftlf_cube if input_format == "scmcube" else None

        result = test_cube.get_area_weights(areacell_scmcube=test_areacella_input)
        test_cube._get_areacell_scmcube.assert_called_with(test_areacella_input)

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "areacell",
        ["not a cube", "cube attr not a cube", "iris_error", "misshaped", "no file"],
    )
    @pytest.mark.parametrize("areacell_var", ["areacella", "areacello", "area_other"])
    @patch(
        "netcdf_scm.iris_cube_wrappers.SCMCube.areacell_var", new_callable=PropertyMock
    )
    def test_get_area_weights_workarounds(
        self,
        mock_areacell_var,
        test_cube,
        test_sftlf_cube,
        areacell_var,
        areacell,
        caplog,
    ):
        mock_areacell_var.return_value = areacell_var

        # can safely ignore these warnings here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*Using DEFAULT_SPHERICAL.*")
            lat_lon_slice = next(
                test_cube.cube.slices([test_cube.lat_name, test_cube.lon_name])
            )
            expected = iris.analysis.cartography.area_weights(lat_lon_slice)

        # we can use test_sftlf_cube here as all we need is an array of the
        # right shape
        if areacell == "iris_error":
            iris_error_msg = "no cube found"
            test_cube.get_metadata_cube = MagicMock(
                side_effect=ConstraintMismatchError(iris_error_msg)
            )
        elif areacell == "misshaped":
            misshaped_cube = SCMCube
            misshaped_cube.cube = iris.cube.Cube(data=np.array([1, 2]))
            misshaped_cube.cube.units = "m**2"
            test_cube.get_metadata_cube = MagicMock(return_value=misshaped_cube)
        elif areacell == "not a cube":
            test_cube.get_metadata_cube = MagicMock(return_value=areacell)
        elif areacell == "cube attr not a cube":
            weird_ob = MagicMock()
            weird_ob.cube = 23
            test_cube.get_metadata_cube = MagicMock(return_value=weird_ob)
        elif areacell == "no file":
            no_file_msg = "No file message here"
            test_cube.get_metadata_cube = MagicMock(side_effect=OSError(no_file_msg))

        caplog.clear()
        caplog.set_level(logging.DEBUG)

        # Function under test
        with pytest.warns(None) as record:
            result = test_cube.get_area_weights()

        test_cube.get_metadata_cube.assert_called_with(areacell_var, cube=None)

        fallback_warn = re.escape(
            "Couldn't find/use areacell_cube, falling back to "
            "iris.analysis.cartography.area_weights"
        )
        radius_warn = re.escape("Using DEFAULT_SPHERICAL_EARTH_RADIUS.")
        if areacell == "iris_error":
            specific_warn = "Could not calculate areacell"
            exc_info = re.escape(iris_error_msg)
        elif areacell == "misshaped":
            specific_warn = "Area weights incompatible with lat lon grid"
            exc_info = re.escape(
                "the sftlf_cube data must be the same shape as the cube's "
                "longitude-latitude grid"
            )
            exc_info = None
        elif areacell == "not a cube":
            specific_warn = "Could not calculate areacell"
            exc_info = re.escape("'str' object has no attribute 'cube'")
        elif areacell == "cube attr not a cube":
            specific_warn = re.escape(
                "areacell cube which was found has cube attribute which isn't an iris cube"
            )
            exc_info = None
        elif areacell == "no file":
            specific_warn = "Could not calculate areacell"
            exc_info = re.escape(no_file_msg)

        assert len(caplog.messages) == 2
        assert re.match(radius_warn, str(record[0].message))
        assert re.match(fallback_warn, str(caplog.messages[1]))
        assert re.match(specific_warn, caplog.messages[0])
        if exc_info is not None:
            assert re.match(
                "Could not calculate areacell, error message: {}".format(exc_info),
                str(caplog.records[0].message),
            )  # the actual message is stored in the exception

        np.testing.assert_array_equal(result, expected)

    @patch("netcdf_scm.iris_cube_wrappers.assert_all_time_axes_same")
    @patch("netcdf_scm.iris_cube_wrappers.get_scm_cube_time_axis_in_calendar")
    @pytest.mark.parametrize("out_calendar", [None, "gregorian", "365_day"])
    def test_get_openscmdata_time_axis_and_calendar(
        self,
        mock_get_time_axis_in_calendar,
        mock_assert_all_time_axes_same,
        test_cube,
        out_calendar,
        caplog,
    ):
        expected_calendar = (
            test_cube.cube.coords("time")[0].units.calendar
            if out_calendar is None
            else out_calendar
        )

        tscm_timeseries_cubes = {"mocked 1": 198, "mocked 2": 248}

        years = range(1920, 1950, 10)
        tget_time_axis = [cftime.DatetimeNoLeap(y, 1, 1) for y in years]
        mock_get_time_axis_in_calendar.return_value = tget_time_axis

        expected_idx = pd.Index(
            [dt.datetime(y, 1, 1) for y in years], dtype="object", name="time"
        )
        result_idx, result_calendar = test_cube._get_openscmdata_time_axis_and_calendar(
            tscm_timeseries_cubes, out_calendar
        )

        assert_index_equal(result_idx, expected_idx)
        assert result_calendar == expected_calendar

        mock_get_time_axis_in_calendar.assert_has_calls(
            [call(c, expected_calendar) for c in tscm_timeseries_cubes.values()],
            any_order=True,
        )
        mock_assert_all_time_axes_same.assert_called_with(
            [tget_time_axis] * len(tscm_timeseries_cubes)
        )

        if out_calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
            assert len(caplog.messages) == 1
            assert caplog.messages[0] == (
                "Performing lazy conversion to datetime for calendar: 365_day. This "
                "may cause subtle errors in operations that depend on the length of "
                "time between dates"
            )
            assert caplog.records[0].levelname == "WARNING"

    @pytest.mark.parametrize(
        "valid_time_period_str",
        [("20150111"), ("2015-2018"), ("20150103-20181228"), ("2015010311-2018122814")],
    )
    def test_check_time_period_valid(self, valid_time_period_str, test_cube):
        test_cube._check_time_period_valid(valid_time_period_str)

    @pytest.mark.parametrize(
        "invalid_time_period_str",
        [
            ("2015011"),
            ("2015-201812"),
            ("201512-201312"),
            ("20150103-201812"),
            ("2015210311-2018122814"),
            ("2015113311-2018122814"),
            ("2015-2018-2019"),
            ("2015210311_2018122814"),
        ],
    )
    def test_check_time_period_invalid(self, invalid_time_period_str, test_cube):
        expected_error_msg = re.escape(
            "Your time_period indicator ({}) does not look right".format(
                invalid_time_period_str
            )
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            test_cube._check_time_period_valid(invalid_time_period_str)

    def test_cube_info(self, test_cube, test_sftlf_cube):
        test_cube._loaded_paths = ["test.nc"]
        test_cube._metadata_cubes = {"sftlf": test_sftlf_cube}
        res = test_cube.info

        assert test_sftlf_cube.info == {
            "files": ["test_sftlf.nc"]  # this value comes from test_sftlf_cube
        }

        exp = {"files": ["test.nc"], "metadata": {"sftlf": test_sftlf_cube.info}}

        assert res == exp

    @pytest.mark.parametrize(
        "realm,expected",
        [
            ("atmos", "areacella"),
            ("ocean", "areacello"),
            ("ocnBgchem", "areacello"),
            ("land", "areacella"),
        ],
    )
    def test_areacell_var(self, test_cube, realm, expected):
        test_cube.cube.attributes[test_cube._realm_key] = realm
        assert test_cube.areacell_var == expected

    def _setup_test_metadata_var(self, test_cube, realm, caplog):
        caplog.set_level(logging.INFO, logger="netcdf_scm.iris_cube_wrappers")
        if realm is None:
            test_cube.cube.attributes.pop(test_cube._realm_key, None)
        else:
            test_cube.cube.attributes[test_cube._realm_key] = realm

        return test_cube

    def _check_metadata_var_test_messages(self, caplog, realm, realm_key="realm"):
        if realm is None:
            assert len(caplog.messages) == 1
            assert caplog.messages[0] == (
                "No `{}` attribute in `self.cube`, NetCDF-SCM will treat the data as "
                "`atmosphere`".format(realm_key)
            )
        elif realm == "junk":
            assert len(caplog.messages) == 1
            assert caplog.messages[0] == (
                "Unrecognised `{}` attribute value, `junk`, in `self.cube`, NetCDF-SCM will treat the data as "
                "`atmosphere`".format(realm_key)
            )
        else:
            assert len(caplog.messages) == 0

    @pytest.mark.parametrize(
        "realm,expected",
        [
            ("atmos", "sftlf"),
            ("ocean", "sftof"),
            ("ocnBgchem", "sftof"),
            ("land", "sftlf"),
            (None, "sftlf"),
        ],
    )
    def test_surface_fraction_var(self, test_cube, realm, expected, caplog):
        test_cube = self._setup_test_metadata_var(test_cube, realm, caplog)
        # do twice to check warning only thrown once
        assert test_cube.surface_fraction_var == expected
        assert test_cube.surface_fraction_var == expected
        self._check_metadata_var_test_messages(
            caplog, realm, realm_key=test_cube._realm_key
        )

    @pytest.mark.parametrize(
        "realm,expected",
        [
            ("atmos", "fx"),
            ("ocean", "Ofx"),
            ("ocnBgchem", "Ofx"),
            ("land", "fx"),
            (None, "fx"),
        ],
    )
    def test_table_name_for_metadata_vars(self, test_cube, realm, expected, caplog):
        test_cube = self._setup_test_metadata_var(test_cube, realm, caplog)
        # do twice to check warning only thrown once
        assert test_cube.table_name_for_metadata_vars == expected
        assert test_cube.table_name_for_metadata_vars == expected
        self._check_metadata_var_test_messages(caplog, realm)

    @pytest.mark.parametrize(
        "realm,expected",
        [
            ("atmos", "atmosphere"),
            ("ocean", "ocean"),
            ("ocnBgchem", "ocean"),
            ("land", "land"),
            (None, "atmosphere"),
            ("junk", "atmosphere"),
        ],
    )
    def test_netcdf_scm_realm(self, test_cube, realm, expected, caplog):
        test_cube = self._setup_test_metadata_var(test_cube, realm, caplog)
        # do twice to check warning only thrown once
        assert test_cube.netcdf_scm_realm == expected
        assert test_cube.netcdf_scm_realm == expected
        self._check_metadata_var_test_messages(caplog, realm)

    def test_lat_lon_shape(self, test_cube):
        assert test_cube.lat_lon_shape == (3, 4)


class _CMIPCubeTester(TestSCMCube):
    tclass = _CMIPCube

    @patch.object(tclass, "load_data_from_identifiers")
    def test_get_metadata_cube(self, mock_load_data_from_identifiers, test_cube):
        tvar = "tmdata_var"
        tload_arg_dict = {"Arg 1": 12, "Arg 2": "Val 2"}

        test_cube._get_metadata_load_arguments = MagicMock(return_value=tload_arg_dict)

        result = test_cube.get_metadata_cube(tvar)

        assert type(result) == type(test_cube)

        test_cube._get_metadata_load_arguments.assert_called_with(tvar)
        assert test_cube._metadata_cubes[tvar] == result
        mock_load_data_from_identifiers.assert_called_with(**tload_arg_dict)

    def test_get_metadata_load_arguments(self, test_cube):
        self.run_test_of_method_to_overload(
            test_cube,
            "_get_metadata_load_arguments",
            junk_args={"metadata_variable": "mdata_var"},
        )

    @pytest.mark.parametrize("process_warnings", [True, False])
    @patch("netcdf_scm.iris_cube_wrappers.iris.load_cube")
    @patch("netcdf_scm.iris_cube_wrappers._check_cube_and_adjust_if_needed")
    def test_load_data_from_identifiers(
        self,
        mock_check_cube_and_adjust_if_needed,
        mock_iris_load_cube,
        test_cube,
        process_warnings,
    ):
        tfile = "hello_world_test.nc"

        test_cube.get_filepath_from_load_data_from_identifiers_args = MagicMock(
            return_value=tfile
        )

        vcons = 12.195
        test_cube.get_variable_constraint = MagicMock(return_value=vcons)

        lcube_return = 9848

        def raise_mock_warn_and_return_test_value(*args, **kwargs):
            warnings.warn("mocked warning")

            return lcube_return

        mock_iris_load_cube.side_effect = raise_mock_warn_and_return_test_value
        mock_check_cube_and_adjust_if_needed.return_value = lcube_return

        test_cube._process_load_data_from_identifiers_warnings = MagicMock()

        tkwargs = {
            "variable_name": "fco2antt",
            "modeling_realm": "Amon",
            "model": "CanESM2",
            "experiment": "1pctCO2",
        }
        test_cube.load_data_from_identifiers(
            process_warnings=process_warnings, **tkwargs
        )

        assert test_cube.cube == lcube_return
        test_cube.get_filepath_from_load_data_from_identifiers_args.assert_called_with(
            **tkwargs
        )
        test_cube.get_variable_constraint.assert_called()
        mock_iris_load_cube.assert_called_with(tfile, constraint=vcons)
        if process_warnings:
            test_cube._process_load_data_from_identifiers_warnings.assert_called()
        else:
            test_cube._process_load_data_from_identifiers_warnings.assert_not_called()

        mock_check_cube_and_adjust_if_needed.assert_called()

        assert test_cube._loaded_paths == [tfile]

    def test_load_missing_variable_error(self, test_cube, test_tas_file):
        tfile = test_tas_file
        test_cube.get_filepath_from_load_data_from_identifiers_args = MagicMock(
            return_value=tfile
        )

        bad_constraint = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str("misnamed_var"))
        )
        test_cube.get_variable_constraint = MagicMock(return_value=bad_constraint)

        with pytest.raises(ConstraintMismatchError, match="no cubes found"):
            test_cube.load_data_from_identifiers(mocked_out="mocked")

    def test_get_filepath_from_load_data_from_identifiers_args(self, test_cube):
        self.run_test_of_method_to_overload(
            test_cube, "get_filepath_from_load_data_from_identifiers_args"
        )

    def test_get_variable_constraint_from_load_data_from_identifiers_args(
        self, test_cube
    ):
        self.run_test_of_method_to_overload(
            test_cube, "get_variable_constraint_from_load_data_from_identifiers_args"
        )

    def test_get_data_directory(self, test_cube):
        self.run_test_of_method_to_overload(test_cube, "get_data_directory")

    def test_get_data_filename(self, test_cube):
        self.run_test_of_method_to_overload(test_cube, "get_data_filename")

    @pytest.mark.parametrize("process_warnings", [True, False])
    def test_load_data_from_path(self, test_cube, process_warnings):
        tpath = "./somewhere/over/the/rainbow/test.nc"
        tids = {"id1": "mocked", "id2": 123}

        test_cube.get_load_data_from_identifiers_args_from_filepath = MagicMock(
            return_value=tids
        )
        test_cube.load_data_from_identifiers = MagicMock()

        test_cube.load_data_from_path(tpath, process_warnings=process_warnings)
        test_cube.get_load_data_from_identifiers_args_from_filepath.assert_called_with(
            tpath
        )
        test_cube.load_data_from_identifiers.assert_called_with(
            process_warnings=process_warnings, **tids
        )

    def test_get_load_data_from_identifiers_args_from_filepath(self, test_cube):
        tpath = "/tpath/file.ext"
        test_cube.process_path = MagicMock(return_value={"a": "b"})
        test_cube.process_filename = MagicMock(return_value={"c": "d"})

        expected = {"a": "b", "c": "d"}
        result = test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)
        assert result == expected

        test_cube.process_path.assert_called_with(dirname(tpath))
        test_cube.process_filename.assert_called_with(basename(tpath))

    def test_get_load_data_from_identifiers_args_from_filepath_errors(self, test_cube):
        tpath = "/tpath/file.ext"
        test_cube.process_path = MagicMock(return_value={"model": "CanESM2"})
        test_cube.process_filename = MagicMock(return_value={"model": "HadGem3"})
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

    def _run_test_add_time_period_from_files_in_directory(
        self, mock_listdir, files_in_path, expected_time_period, test_cube
    ):
        mock_listdir.return_value = files_in_path
        tdir = "mocked"
        test_cube._check_data_names_in_same_directory = MagicMock()

        test_cube._add_time_period_from_files_in_directory(tdir)

        assert test_cube._time_id == expected_time_period
        test_cube._check_data_names_in_same_directory.assert_called_with(tdir)

    def _run_test_get_filepath_from_load_data_from_identifiers_args(
        self, test_cube, tkwargs_list
    ):
        tkwargs = {k: getattr(self, "t" + k) for k in tkwargs_list}

        mock_data_path = "here/there/everywhere"
        test_cube.get_data_directory = MagicMock(return_value=mock_data_path)

        mock_data_name = "here_there_file.nc"
        test_cube.get_data_filename = MagicMock(return_value=mock_data_name)

        for kwarg in tkwargs_list:
            if hasattr(test_cube, kwarg) and getattr(test_cube, kwarg) is None:
                continue  # temporary hack in middle of refactoring
            with pytest.raises(AttributeError):
                getattr(test_cube, kwarg)

        result = test_cube.get_filepath_from_load_data_from_identifiers_args(**tkwargs)
        expected = join(mock_data_path, mock_data_name)

        assert result == expected

        for kwarg in tkwargs_list:
            assert getattr(test_cube, kwarg) == tkwargs[kwarg]

        assert test_cube.get_data_directory.call_count == 1
        assert test_cube.get_data_filename.call_count == 1

    def test_get_data_directory_error(self):
        inst = self.tclass()
        error_msg = re.escape("Could not determine data directory")
        with pytest.raises(OSError, match=error_msg):
            inst.get_data_directory()

    def test_get_data_filename_error(self):
        inst = self.tclass()
        error_msg = re.escape("Could not determine data filename")
        with pytest.raises(OSError, match=error_msg):
            inst.get_data_filename()


class TestMarbleCMIP5Cube(_CMIPCubeTester):
    tclass = MarbleCMIP5Cube
    tactivity = "cmip5"
    texperiment = "1pctCO2"
    tmip_table = "Amon"
    tvariable_name = "tas"
    tmodel = "CanESM2"
    tensemble_member = "r1i1p1"
    ttime_period = "185001-198912"
    tfile_ext = ".nc"
    attributes_to_set_from_fixtures = {"troot_dir": "test_data_marble_cmip5_dir"}

    @patch("netcdf_scm.iris_cube_wrappers.os.listdir")
    @pytest.mark.parametrize(
        "files_in_path, expected_time_period",
        [
            (
                [
                    "tas_Amon_HadCM3_rcp45_r1i1p1_203601-203812.nc",
                    "tas_Amon_HadCM3_rcp45_r1i1p1_200601-203012.nc",
                    "tas_Amon_HadCM3_rcp45_r1i1p1_203101-203512.nc",
                ],
                "200601-203812",
            ),
            (
                [
                    "tas_Amon_HadCM3_rcp45_r1i1p1_103601-103812.nc",
                    "tas_Amon_HadCM3_rcp45_r1i1p1_003101-103512.nc",
                    "tas_Amon_HadCM3_rcp45_r1i1p1_000601-003012.nc",
                ],
                "000601-103812",
            ),
        ],
    )
    def test_add_time_period_from_files_in_directory(
        self, mock_listdir, files_in_path, expected_time_period, test_cube
    ):
        self._run_test_add_time_period_from_files_in_directory(
            mock_listdir, files_in_path, expected_time_period, test_cube
        )
        assert test_cube.time_period == expected_time_period

    def test_get_filepath_from_load_data_from_identifiers_args(self, test_cube):
        tkwargs_list = [
            "root_dir",
            "activity",
            "experiment",
            "mip_table",
            "variable_name",
            "model",
            "ensemble_member",
            "time_period",
            "file_ext",
        ]
        self._run_test_get_filepath_from_load_data_from_identifiers_args(
            test_cube, tkwargs_list
        )

    def test_process_filename(self, test_cube):
        tname = "fco2antt_Amon_CanESM2_1pctCO2_r1i1p1_185001-198912.nc"
        result = test_cube.process_filename(tname)
        expected = {
            "experiment": "1pctCO2",
            "mip_table": "Amon",
            "variable_name": "fco2antt",
            "model": "CanESM2",
            "ensemble_member": "r1i1p1",
            "time_period": "185001-198912",
            "file_ext": ".nc",
        }

        assert result == expected

    @pytest.mark.parametrize(
        "tname",
        [
            "sftlf_CanESM2_1pctCO2_r0i0p0.nc",
            "sftlf_fx_CanESM2_1pctCO2.nc",
            "sftlf_fx_CanESM2_1pctCO2-r0i0p0.nc",
            "fco2antt_Amon_CanESM2_1pctCO2_r1i1p1_.nc",
        ],
    )
    def test_process_filename_errors(self, test_cube, tname):
        error_msg = re.escape("Filename does not look right: {}".format(tname))
        with pytest.raises(ValueError, match=error_msg):
            test_cube.process_filename(tname)

    def test_process_path(self, test_cube):
        tpath = (
            "/tests/test_data/marble_cmip5/cmip5/1pctCO2/Amon/fco2antt/CanESM2/r1i1p1/"
        )
        result = test_cube.process_path(tpath)
        expected = {
            "root_dir": "/tests/test_data/marble_cmip5",
            "activity": "cmip5",
            "experiment": "1pctCO2",
            "mip_table": "Amon",
            "variable_name": "fco2antt",
            "model": "CanESM2",
            "ensemble_member": "r1i1p1",
        }

        assert result == expected

    @pytest.mark.parametrize(
        "tpath",
        [
            "1pctCO2/fx/sftlf/CanESM2/r0i0p0",
            "cmip5/1pctCO2/sftlf/CanESM2/r0i0p0",
            "cmip5/1pctCO2_today/fx/sftlf/CanESM2/r0i0p0",
        ],
    )
    def test_process_path_errors(self, test_cube, tpath):
        error_msg = re.escape("Path does not look right: {}".format(tpath))
        with pytest.raises(ValueError, match=error_msg):
            test_cube.process_path(tpath)

    def test_get_data_directory(self, test_cube):
        expected = join(
            self.troot_dir,
            self.tactivity,
            self.texperiment,
            self.tmip_table,
            self.tvariable_name,
            self.tmodel,
            self.tensemble_member,
        )

        atts_to_set = [
            "root_dir",
            "activity",
            "experiment",
            "mip_table",
            "variable_name",
            "model",
            "ensemble_member",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        result = test_cube.get_data_directory()

        assert result == expected

    def test_get_data_filename(self, test_cube):
        expected = (
            "_".join(
                [
                    self.tvariable_name,
                    self.tmip_table,
                    self.tmodel,
                    self.texperiment,
                    self.tensemble_member,
                    self.ttime_period,
                ]
            )
            + self.tfile_ext
        )

        atts_to_set = [
            "experiment",
            "mip_table",
            "variable_name",
            "model",
            "ensemble_member",
            "time_period",
            "file_ext",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        result = test_cube.get_data_filename()

        assert result == expected

    def test_get_data_filename_no_time(self, test_cube):
        expected = (
            "_".join(
                [
                    self.tvariable_name,
                    self.tmip_table,
                    self.tmodel,
                    self.texperiment,
                    self.tensemble_member,
                ]
            )
            + self.tfile_ext
        )

        atts_to_set = [
            "experiment",
            "mip_table",
            "variable_name",
            "model",
            "ensemble_member",
            "file_ext",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        test_cube.time_period = None

        result = test_cube.get_data_filename()

        assert result == expected

    def test_get_variable_constraint_from_load_data_from_identifiers_args(
        self, test_cube
    ):
        tkwargs_list = [
            "root_dir",
            "activity",
            "experiment",
            "mip_table",
            "variable_name",
            "model",
            "ensemble_member",
            "time_period",
            "file_ext",
        ]
        tkwargs = {k: getattr(self, "t" + k) for k in tkwargs_list}

        # impossible to do other tests as far as I can tell because you have to pass a
        # local function in both the test and the argument, help welcome. expected is
        # here.
        """
        expected = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str(tkwargs["variable_name"]))
        )
        """
        for k, v in tkwargs.items():
            setattr(test_cube, k, v)
        result = test_cube.get_variable_constraint()
        assert isinstance(result, iris.Constraint)

    def test_get_metadata_load_arguments(self, test_cube):
        tmetadata_var = "mdata_var"
        atts_to_set = [
            "root_dir",
            "activity",
            "experiment",
            "mip_table",
            "variable_name",
            "model",
            "ensemble_member",
            "time_period",
            "file_ext",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        expected = {
            "root_dir": test_cube.root_dir,
            "activity": test_cube.activity,
            "experiment": test_cube.experiment,
            "mip_table": "fx",
            "variable_name": tmetadata_var,
            "model": test_cube.model,
            "ensemble_member": "r0i0p0",
            "time_period": None,
            "file_ext": test_cube.file_ext,
        }

        result = test_cube._get_metadata_load_arguments(tmetadata_var)

        assert result == expected

    def test_get_scm_timeseries_ids_warnings(
        self, test_cube, caplog, expected_mip_era="CMIP5", expected_warns=5
    ):
        super().test_get_scm_timeseries_ids_warnings(
            test_cube, caplog, expected_mip_era="CMIP5", expected_warns=5
        )

    @pytest.mark.parametrize(
        "realm,expected",
        [
            ("atmos", "fx"),
            ("ocean", "fx"),
            ("ocnBgchem", "fx"),
            ("land", "fx"),
            (None, "fx"),
        ],
    )
    def test_table_name_for_metadata_vars(self, test_cube, realm, expected, caplog):
        test_cube = self._setup_test_metadata_var(test_cube, realm, caplog)
        # do twice to check warning only thrown once
        assert test_cube.table_name_for_metadata_vars == expected
        assert test_cube.table_name_for_metadata_vars == expected

    @pytest.mark.parametrize(
        "realm,expected",
        [
            ("atmos", "atmosphere"),
            ("ocean", "ocean"),
            ("ocnBgchem", "ocean"),
            ("land", "land"),
            (None, "atmosphere"),
        ],
    )
    def test_netcdf_scm_realm(self, test_cube, realm, expected, caplog):
        test_cube = self._setup_test_metadata_var(test_cube, realm, caplog)
        # do twice to check warning only thrown once
        assert test_cube.netcdf_scm_realm == expected
        assert test_cube.netcdf_scm_realm == expected
        self._check_metadata_var_test_messages(
            caplog, realm, realm_key="modeling_realm"
        )


class TestCMIP6Input4MIPsCube(_CMIPCubeTester):
    tclass = CMIP6Input4MIPsCube
    troot_dir = "cmip6input4mipstestdata"
    tactivity_id = "input4MIPs"
    tmip_era = "CMIP6"
    ttarget_mip = "CMIP"
    tinstitution_id = "PCMDI"
    tsource_id = "PCMDI-AMIP-1-1-4"
    trealm = "ocean"
    tfrequency = "mon"
    tvariable_id = "tos"
    tgrid_label = "gn"
    tversion = "v20180427"
    tdataset_category = "SSTsAndSeaIce"
    ttime_range = "2015-2100"
    tfile_ext = ".nc"

    @patch("netcdf_scm.iris_cube_wrappers.os.listdir")
    @pytest.mark.parametrize(
        "files_in_path, expected_time_period",
        [
            (
                [
                    "multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_200601-201012.nc",
                    "multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_201401-203812.nc",
                    "multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_201101-201312.nc",
                ],
                "200601-203812",
            ),
            (
                [
                    "multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_010009-103812.nc",
                    "multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_000601-009912.nc",
                    "multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_010001-010008.nc",
                ],
                "000601-103812",
            ),
        ],
    )
    def test_add_time_period_from_files_in_directory(
        self, mock_listdir, files_in_path, expected_time_period, test_cube
    ):
        self._run_test_add_time_period_from_files_in_directory(
            mock_listdir, files_in_path, expected_time_period, test_cube
        )
        assert test_cube.time_range == expected_time_period

    def test_get_filepath_from_load_data_from_identifiers_args(self, test_cube):
        test_cube._check_self_consistency = MagicMock()

        tkwargs_list = [
            "root_dir",
            "activity_id",
            "mip_era",
            "target_mip",
            "institution_id",
            "source_id",
            "realm",
            "frequency",
            "variable_id",
            "grid_label",
            "version",
            "dataset_category",
            "time_range",
            "file_ext",
        ]
        self._run_test_get_filepath_from_load_data_from_identifiers_args(
            test_cube, tkwargs_list
        )

        test_cube._check_self_consistency.assert_called()

    def test_process_filename(self, test_cube):
        tname = "tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc"
        result = test_cube.process_filename(tname)
        expected = {
            "activity_id": "input4MIPs",
            "target_mip": "CMIP",
            "source_id": "PCMDI-AMIP-1-1-4",
            "variable_id": "tos",
            "grid_label": "gn",
            "dataset_category": "SSTsAndSeaIce",
            "time_range": "187001-201712",
            "file_ext": ".nc",
        }

        assert result == expected

    @pytest.mark.parametrize(
        "tname",
        [
            "tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_.nc",
            "tos_input4MIPs_PCMDI-AMIP-1-1-4_gn_187001-201712.nc",
            "input4MIPs_SSTsAndSeaIce_PCMDI-AMIP-1-1-4_gn_187001-201712.nc",
        ],
    )
    def test_process_filename_errors(self, test_cube, tname):
        error_msg = re.escape("Filename does not look right: {}".format(tname))
        with pytest.raises(ValueError, match=error_msg):
            test_cube.process_filename(tname)

    def test_process_path(self, test_cube):
        tpath = "/tests/test_data/cmip6-input4mips/input4MIPs/CMIP6/CMIP/PCMDI/PCMDI-AMIP-1-1-4/ocean/mon/tos/gn/v20180427/"
        result = test_cube.process_path(tpath)
        expected = {
            "root_dir": "/tests/test_data/cmip6-input4mips",
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
        }
        assert result == expected

    @pytest.mark.parametrize(
        "tpath",
        [
            "input4MIPs/CMIP6/CMIP/PCMDI/PCMDI-AMIP-1-1-4/ocean/mon/tos/gn",
            "input4MIPs/CMIP6/CMIP/PCMDI-AMIP-1-1-4/ocean/mon/tos/gn/v20180427",
            "input4MIPs/CMIP6/CMIP/PCMDI/PCMDI-AMIP-1-1-4/ocean/tos/gn/v20180427",
        ],
    )
    def test_process_path_errors(self, test_cube, tpath):
        error_msg = re.escape("Path does not look right: {}".format(tpath))
        with pytest.raises(ValueError, match=error_msg):
            test_cube.process_path(tpath)

    def test_check_self_consistency(self, test_cube):
        test_cube.source_id = "UoM-REMIND-MAGPIE-ssp585-1-2-0"
        test_cube.institution_id = "UoB"

        error_msg = re.escape("source_id must contain institution_id")
        with pytest.raises(AssertionError, match=error_msg):
            test_cube._check_self_consistency()

    def test_get_metadata_load_arguments(self, test_cube):
        tmetadata_var = "mdata_var"
        atts_to_set = [
            "root_dir",
            "activity_id",
            "mip_era",
            "target_mip",
            "institution_id",
            "source_id",
            "realm",
            "frequency",
            "variable_id",
            "grid_label",
            "version",
            "dataset_category",
            "time_range",
            "file_ext",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        expected = {
            "root_dir": test_cube.root_dir,
            "activity_id": test_cube.activity_id,
            "mip_era": test_cube.mip_era,
            "target_mip": test_cube.target_mip,
            "institution_id": test_cube.institution_id,
            "source_id": test_cube.source_id,
            "realm": test_cube.realm,
            "frequency": "fx",
            "variable_id": tmetadata_var,
            "grid_label": test_cube.grid_label,
            "version": test_cube.version,
            "dataset_category": test_cube.dataset_category,
            "time_range": None,
            "file_ext": test_cube.file_ext,
        }

        result = test_cube._get_metadata_load_arguments(tmetadata_var)

        assert result == expected

    def test_get_data_filename(self, test_cube):
        expected = (
            "_".join(
                [
                    self.tvariable_id,
                    self.tactivity_id,
                    self.tdataset_category,
                    self.ttarget_mip,
                    self.tsource_id,
                    self.tgrid_label,
                ]
            )
            + self.tfile_ext
        )

        test_cube.time_range = None
        atts_to_set = [
            "variable_id",
            "activity_id",
            "dataset_category",
            "target_mip",
            "source_id",
            "grid_label",
            "file_ext",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        result = test_cube.get_data_filename()
        assert result == expected

        expected = (
            "_".join(
                [
                    self.tvariable_id,
                    self.tactivity_id,
                    self.tdataset_category,
                    self.ttarget_mip,
                    self.tsource_id,
                    self.tgrid_label,
                    self.ttime_range,
                ]
            )
            + self.tfile_ext
        )

        atts_to_set = ["time_range"]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        result = test_cube.get_data_filename()
        assert result == expected

    def test_get_data_directory(self, test_cube):
        expected = join(
            self.troot_dir,
            self.tactivity_id,
            self.tmip_era,
            self.ttarget_mip,
            self.tinstitution_id,
            self.tsource_id,
            self.trealm,
            self.tfrequency,
            self.tvariable_id,
            self.tgrid_label,
            self.tversion,
        )

        atts_to_set = [
            "root_dir",
            "activity_id",
            "mip_era",
            "target_mip",
            "institution_id",
            "source_id",
            "realm",
            "frequency",
            "variable_id",
            "grid_label",
            "version",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        result = test_cube.get_data_directory()

        assert result == expected

    def test_get_variable_constraint_from_load_data_from_identifiers_args(
        self, test_cube
    ):
        # TODO and refactor to make attribute of self
        tkwargs_list = [
            "root_dir",
            "activity_id",
            "mip_era",
            "target_mip",
            "institution_id",
            "source_id",
            "realm",
            "frequency",
            "variable_id",
            "grid_label",
            "version",
            "dataset_category",
            "time_range",
            "file_ext",
        ]
        tkwargs = {k: getattr(self, "t" + k) for k in tkwargs_list}

        # impossible to do other tests as far as I can tell because you have to pass a
        # local function in both the test and the argument, help welcome. expected is
        # here.
        """
        expected = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str(tkwargs["variable_name"]))
        )
        """
        for k, v in tkwargs.items():
            setattr(test_cube, k, v)
        result = test_cube.get_variable_constraint()
        assert isinstance(result, iris.Constraint)


class TestCMIP6OutputCube(_CMIPCubeTester):
    tclass = CMIP6OutputCube
    troot_dir = "cmip6input4mipstestdata"
    tmip_era = "CMIP6"
    tactivity_id = "DCPP"
    tinstitution_id = "CNRM-CERFACS"
    tsource_id = "CNRM-CM6-1"
    texperiment_id = "dcppA-hindcast"
    tmember_id = "s1960-r2i1p1f3"
    ttable_id = "day"
    tvariable_id = "pr"
    tgrid_label = "gn"
    tversion = "v20160215"
    ttime_range = "198001-198412"
    tfile_ext = ".nc"

    @patch("netcdf_scm.iris_cube_wrappers.os.listdir")
    @pytest.mark.parametrize(
        "files_in_path, expected_time_period",
        [
            (
                [
                    "pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f1_gn_200601-201012.nc",
                    "pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f1_gn_201401-203812.nc",
                    "pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f1_gn_201101-201312.nc",
                ],
                "200601-203812",
            ),
            (
                [
                    "pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f1_gn_010009-103812.nc",
                    "pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f1_gn_000601-009912.nc",
                    "pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f1_gn_010001-010008.nc",
                ],
                "000601-103812",
            ),
        ],
    )
    def test_add_time_period_from_files_in_directory(
        self, mock_listdir, files_in_path, expected_time_period, test_cube
    ):
        self._run_test_add_time_period_from_files_in_directory(
            mock_listdir, files_in_path, expected_time_period, test_cube
        )
        assert test_cube.time_range == expected_time_period

    def test_get_filepath_from_load_data_from_identifiers_args(self, test_cube):
        tkwargs_list = [
            "root_dir",
            "mip_era",
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "member_id",
            "table_id",
            "variable_id",
            "grid_label",
            "version",
            "time_range",
            "file_ext",
        ]
        self._run_test_get_filepath_from_load_data_from_identifiers_args(
            test_cube, tkwargs_list
        )

    def test_process_filename(self, test_cube):
        tname = "pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f3_gn_198001-198412.nc"
        result = test_cube.process_filename(tname)
        expected = {
            "source_id": "CNRM-CM6-1",
            "experiment_id": "dcppA-hindcast",
            "member_id": "s1960-r2i1p1f3",
            "variable_id": "pr",
            "table_id": "day",
            "grid_label": "gn",
            "time_range": "198001-198412",
            "file_ext": ".nc",
        }

        assert result == expected

    @pytest.mark.parametrize(
        "tname",
        [
            "pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f3_gn_.nc",
            "dcppA-hindcast_s1960-r2i1p1f3_gn_198001-198412.nc"
            "pr_day_CNRM-CM6-1_dcppA-hindcast_s1960-r2i1p1f3.nc",
        ],
    )
    def test_process_filename_errors(self, test_cube, tname):
        error_msg = re.escape("Filename does not look right: {}".format(tname))
        with pytest.raises(ValueError, match=error_msg):
            test_cube.process_filename(tname)

    def test_process_path(self, test_cube):
        tpath = "CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/s1960-r2i1p1f3/day/pr/gn/v20160215/"
        result = test_cube.process_path(tpath)
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
        }

        assert result == expected

    @pytest.mark.parametrize(
        "tpath",
        [
            "/DCPP/CNRM-CERFACS/CNRM-CM6-1/dcppA-hindcast/day/pr/gn/v20160215",
            "CMIP6/DCPP/CNRM-CERFACS/CNRM-CM6-1/s1960-r2i1p1f3/day/pr/gn/v20160215",
            "CMIP6/DCPP/CNRM-CM6-1/dcppA-hindcast/s1960-r2i1p1f3/pr/gn/v20160215",
        ],
    )
    def test_process_path_errors(self, test_cube, tpath):
        error_msg = re.escape("Path does not look right: {}".format(tpath))
        with pytest.raises(ValueError, match=error_msg):
            test_cube.process_path(tpath)

    def test_get_metadata_load_arguments(self, test_cube):
        tmetadata_var = "mdata_var"
        atts_to_set = [
            "root_dir",
            "mip_era",
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "member_id",
            "table_id",
            "variable_id",
            "grid_label",
            "version",
            "time_range",
            "file_ext",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        expected = {
            "root_dir": test_cube.root_dir,
            "mip_era": test_cube.mip_era,
            "activity_id": test_cube.activity_id,
            "institution_id": test_cube.institution_id,
            "source_id": test_cube.source_id,
            "experiment_id": test_cube.experiment_id,
            "member_id": test_cube.member_id,
            "table_id": "fx",
            "variable_id": tmetadata_var,
            "grid_label": test_cube.grid_label,
            "version": test_cube.version,
            "time_range": None,
            "file_ext": test_cube.file_ext,
        }

        result = test_cube._get_metadata_load_arguments(tmetadata_var)

        assert result == expected

    def test_get_data_filename(self, test_cube):
        expected = (
            "_".join(
                [
                    self.tvariable_id,
                    self.ttable_id,
                    self.tsource_id,
                    self.texperiment_id,
                    self.tmember_id,
                    self.tgrid_label,
                ]
            )
            + self.tfile_ext
        )

        test_cube.time_range = None
        atts_to_set = [
            "source_id",
            "experiment_id",
            "member_id",
            "table_id",
            "variable_id",
            "grid_label",
            "file_ext",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        result = test_cube.get_data_filename()
        assert result == expected

        expected = (
            "_".join(
                [
                    self.tvariable_id,
                    self.ttable_id,
                    self.tsource_id,
                    self.texperiment_id,
                    self.tmember_id,
                    self.tgrid_label,
                    self.ttime_range,
                ]
            )
            + self.tfile_ext
        )

        atts_to_set = ["time_range"]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        result = test_cube.get_data_filename()
        assert result == expected

    def test_get_data_directory(self, test_cube):
        expected = join(
            self.troot_dir,
            self.tmip_era,
            self.tactivity_id,
            self.tinstitution_id,
            self.tsource_id,
            self.texperiment_id,
            self.tmember_id,
            self.ttable_id,
            self.tvariable_id,
            self.tgrid_label,
            self.tversion,
        )

        atts_to_set = [
            "root_dir",
            "mip_era",
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "member_id",
            "table_id",
            "variable_id",
            "grid_label",
            "version",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        result = test_cube.get_data_directory()

        assert result == expected

    def test_get_variable_constraint_from_load_data_from_identifiers_args(
        self, test_cube
    ):
        # TODO and refactor to make attribute of self
        tkwargs_list = [
            "root_dir",
            "mip_era",
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "member_id",
            "table_id",
            "variable_id",
            "grid_label",
            "version",
            "time_range",
            "file_ext",
        ]
        tkwargs = {k: getattr(self, "t" + k) for k in tkwargs_list}

        # impossible to do other tests as far as I can tell because you have to pass a
        # local function in both the test and the argument, help welcome. expected is
        # here.
        """
        expected = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str(tkwargs["variable_name"]))
        )
        """
        for k, v in tkwargs.items():
            setattr(test_cube, k, v)
        result = test_cube.get_variable_constraint()
        assert isinstance(result, iris.Constraint)

    def test_get_scm_timeseries_ids(self, test_cube, caplog):
        tmodel = "ABCD"
        tactivity = "rcpmip"
        tensemble_member = "r1i3p10"
        tscenario = "oscvolcanicrf"
        tmip_era = "CMIP6test"
        test_cube.source_id = tmodel
        test_cube.activity_id = tactivity
        test_cube.experiment_id = tscenario
        test_cube.member_id = tensemble_member
        test_cube.mip_era = tmip_era

        res = test_cube._get_scm_timeseries_ids()
        assert res["climate_model"] == tmodel
        assert res["scenario"] == tscenario
        assert res["activity_id"] == tactivity
        assert res["member_id"] == tensemble_member
        assert res["mip_era"] == tmip_era
