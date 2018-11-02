from os.path import join
from unittest.mock import patch, MagicMock, call
import warnings
import itertools


import pytest
import re
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import iris
from iris.exceptions import ConstraintMismatchError
from iris.util import broadcast_to_shape


from netcdf_scm.iris_cube_wrappers import SCMCube, MarbleCMIP5Cube
from conftest import (
    TEST_DATA_MARBLE_CMIP5_DIR,
    TEST_TAS_FILE,
    TEST_AREACELLA_FILE,
    tdata_required,
)


class TestSCMCube(object):
    tclass = SCMCube

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
    def test_load_data_from_identifiers(self, mock_iris_load_cube, test_cube):
        tfile = "hello_world_test.nc"
        test_cube.get_filepath_from_load_data_from_identifiers_args = MagicMock(
            return_value=tfile
        )

        vcons = 12.195
        test_cube.get_variable_constraint_from_load_data_from_identifiers_args = MagicMock(
            return_value=vcons
        )

        lcube_return = 9848
        mock_iris_load_cube.return_value = lcube_return

        test_cube._process_load_data_from_identifiers_warnings = MagicMock()

        tkwargs = {
            "variable_name": "fco2antt",
            "modeling_realm": "Amon",
            "model": "CanESM2",
            "experiment": "1pctCO2",
        }
        test_cube.load_data_from_identifiers(**tkwargs)

        assert test_cube.cube == lcube_return
        test_cube.get_filepath_from_load_data_from_identifiers_args.assert_called_with(
            **tkwargs
        )
        test_cube.get_variable_constraint_from_load_data_from_identifiers_args.assert_called_with(
            **tkwargs
        )
        mock_iris_load_cube.assert_called_with(tfile, vcons)
        test_cube._process_load_data_from_identifiers_warnings.assert_not_called()

    def test_process_load_data_from_identifiers_warnings(self, test_cube):
        warn_1 = "warning 1"
        warn_2 = "warning 2"
        warn_area = (
            "PATH-STAMP Missing CF-netCDF measure variable 'areacella' other stuff"
        )
        with warnings.catch_warnings(record=True) as mock_warn_no_area:
            warnings.warn(warn_1)
            warnings.warn(warn_2)

        with warnings.catch_warnings(record=True) as mock_warn_no_area_result:
            test_cube._process_load_data_from_identifiers_warnings(mock_warn_no_area)

        assert len(mock_warn_no_area_result) == 2  # just rethrow warnings
        assert str(mock_warn_no_area_result[0].message) == warn_1
        assert str(mock_warn_no_area_result[1].message) == warn_2

        with warnings.catch_warnings(record=True) as mock_warn_area:
            warnings.warn(warn_1)
            warnings.warn(warn_2)
            warnings.warn(warn_area)

        with warnings.catch_warnings(record=True) as mock_warn_area_result:
            test_cube._process_load_data_from_identifiers_warnings(mock_warn_area)

        assert len(mock_warn_area_result) == 4  # warnings plus extra one
        assert str(mock_warn_area_result[0].message) == warn_1
        assert str(mock_warn_area_result[1].message) == warn_2
        assert "Tried to add areacella cube, failed" in str(
            mock_warn_area_result[2].message
        )
        assert "Missing CF-netCDF measure variable" in str(
            mock_warn_area_result[3].message
        )

    @tdata_required
    def test_add_areacella_measure(self, test_cube):
        # can safely ignore warnings here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*Missing CF-netCDF measure.*")
            test_cube.cube = iris.load_cube(TEST_TAS_FILE)

        tareacellacube = type(test_cube)()

        tareacellacube.cube = iris.load_cube(TEST_AREACELLA_FILE)
        test_cube.get_metadata_cube = MagicMock(return_value=tareacellacube)

        test_cube._add_areacella_measure()

        assert any(["area" in cm.measure for cm in test_cube.cube.cell_measures()])
        assert any(
            ["cell_area" in cm.standard_name for cm in test_cube.cube.cell_measures()]
        )

    @tdata_required
    def test_load_missing_variable_error(self, test_cube):
        tfile = TEST_TAS_FILE
        test_cube.get_filepath_from_load_data_from_identifiers_args = MagicMock(
            return_value=tfile
        )

        bad_constraint = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str("misnamed_var"))
        )
        test_cube.get_variable_constraint_from_load_data_from_identifiers_args = MagicMock(
            return_value=bad_constraint
        )

        with pytest.raises(ConstraintMismatchError, match="no cubes found"):
            test_cube.load_data_from_identifiers(mocked_out="mocked")

    @patch("netcdf_scm.iris_cube_wrappers.iris.load_cube")
    def test_load_data_from_path(self, mock_iris_load_cube, test_cube):
        if type(test_cube) is SCMCube:
            tpath = "here/there/everywehre/test.nc"
            test_cube.load_data_from_path(tpath)
            mock_iris_load_cube.assert_called_with(tpath)
        else:
            self.raise_overload_message("test_load_data")

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

    def test_load_data_in_directory(self, test_cube):
        tdir = "mocked/out"

        test_cube._check_data_names_in_same_directory = MagicMock()
        test_cube._load_and_concatenate_files_in_directory = MagicMock()

        test_cube.load_data_in_directory(tdir)

        test_cube._check_data_names_in_same_directory.assert_called_with(tdir)
        test_cube._load_and_concatenate_files_in_directory.assert_called_with(tdir)

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

    def test_get_data_directory(self, test_cube):
        self.run_test_of_method_to_overload(test_cube, "_get_data_directory")

    def test_get_data_filename(self, test_cube):
        self.run_test_of_method_to_overload(test_cube, "_get_data_filename")

    def test_get_load_data_from_identifiers_args_from_filepath(self, test_cube):
        self.run_test_of_method_to_overload(
            test_cube, "get_load_data_from_identifiers_args_from_filepath"
        )

    @patch.object(tclass, "load_data_from_identifiers")
    def test_get_metadata_cube(self, mock_load_data_from_identifiers, test_cube):
        tvar = "tmdata_var"
        tload_arg_dict = {"Arg 1": 12, "Arg 2": "Val 2"}

        test_cube._get_metadata_load_arguments = MagicMock(return_value=tload_arg_dict)

        result = test_cube.get_metadata_cube(tvar)

        assert type(result) == type(test_cube)

        test_cube._get_metadata_load_arguments.assert_called_with(tvar)
        mock_load_data_from_identifiers.assert_called_with(**tload_arg_dict)

    def test_get_metadata_load_arguments(self, test_cube):
        self.run_test_of_method_to_overload(
            test_cube,
            "_get_metadata_load_arguments",
            junk_args={"metadata_variable": "mdata_var"},
        )

    def test_get_scm_timeseries(self, test_sftlf_cube, test_cube):
        tsftlf_cube = "mocked 124"
        tland_mask_threshold = "mocked 51"
        tareacella_scmcube = "mocked 4389"

        test_cubes_return = {
            "World|Northern Hemisphere|Ocean": 4,
            "World|Southern Hemisphere|Land": 12,
        }
        test_cube.get_scm_timeseries_cubes = MagicMock(return_value=test_cubes_return)

        test_conversion_return = pd.DataFrame(data=np.array([1, 2, 3]))
        test_cube._convert_scm_timeseries_cubes_to_openscmdata = MagicMock(
            return_value=test_conversion_return
        )

        result = test_cube.get_scm_timeseries(
            sftlf_cube=tsftlf_cube,
            land_mask_threshold=tland_mask_threshold,
            areacella_scmcube=tareacella_scmcube,
        )

        test_cube.get_scm_timeseries_cubes.assert_called_with(
            sftlf_cube=tsftlf_cube,
            land_mask_threshold=tland_mask_threshold,
            areacella_scmcube=tareacella_scmcube,
        )
        test_cube._convert_scm_timeseries_cubes_to_openscmdata.assert_called_with(
            test_cubes_return
        )

        assert_frame_equal(result, test_conversion_return)

    def test_get_model_scenario(self, test_cube):
        warn_msg = (
            "Could not determine appropriate model scenario combination, filling with "
            "'unknown'"
        )
        with warnings.catch_warnings(record=True) as recorded_warnings:
            model, scenario = test_cube._get_model_scenario()

        assert model == "unknown"
        assert scenario == "unknown"
        assert len(recorded_warnings) == 1
        assert str(recorded_warnings[0].message) == warn_msg

        tmodel = "ABCD"
        tactivity = "rcpmip"
        texperiment = "oscvolcanicrf"
        tensemble_member = "r1i3p10"
        tscenario = "_".join([tactivity, texperiment, tensemble_member])
        test_cube.model = tmodel
        test_cube.activity = tactivity
        test_cube.experiment = texperiment
        test_cube.ensemble_member = tensemble_member

        model, scenario = test_cube._get_model_scenario()
        assert model == tmodel
        assert scenario == tscenario

    @patch("netcdf_scm.iris_cube_wrappers.take_lat_lon_mean")
    def test_get_scm_timeseries_cubes(self, mock_take_lat_lon_mean, test_cube):
        tsftlf_cube = "mocked out"
        tland_mask_threshold = 48
        tareacella_scmcube = "mocked out again"

        tarea_weights = 145
        test_cube._get_area_weights = MagicMock(return_value=tarea_weights)

        tscm_cubes = {"mock 1": 213, "mock 2": 5893}
        test_cube.get_scm_cubes = MagicMock(return_value=tscm_cubes)

        tlat_lon_mean = "hello mock"
        mock_take_lat_lon_mean.return_value = tlat_lon_mean

        expected = {k: tlat_lon_mean for k in tscm_cubes}
        result = test_cube.get_scm_timeseries_cubes(
            tsftlf_cube, tland_mask_threshold, tareacella_scmcube
        )

        assert result == expected

        test_cube._get_area_weights.assert_called_with(
            areacella_scmcube=tareacella_scmcube
        )
        test_cube.get_scm_cubes.assert_called_with(
            sftlf_cube=tsftlf_cube, land_mask_threshold=tland_mask_threshold
        )
        mock_take_lat_lon_mean.call_count == len(tscm_cubes)

        expected_calls = itertools.product(tscm_cubes.values(), [tarea_weights])
        mock_take_lat_lon_mean.assert_has_calls(
            [call(*c) for c in expected_calls], any_order=True
        )

    @patch("netcdf_scm.iris_cube_wrappers.apply_mask")
    def test_get_scm_cubes(self, mock_apply_mask, test_cube):
        tsftlf_cube = "mocked out"
        tland_mask_threshold = 48

        tscm_masks = {"mask 1": 12, "mask 2": 83}
        test_cube._get_scm_masks = MagicMock(return_value=tscm_masks)

        tapply_mask = 3.14
        mock_apply_mask.return_value = tapply_mask

        expected = {k: tapply_mask for k in tscm_masks}
        result = test_cube.get_scm_cubes(
            sftlf_cube=tsftlf_cube, land_mask_threshold=tland_mask_threshold
        )

        assert result == expected
        test_cube._get_scm_masks.assert_called_with(
            sftlf_cube=tsftlf_cube, land_mask_threshold=tland_mask_threshold
        )

        mock_apply_mask.call_count == len(tscm_masks)
        mock_apply_mask.assert_has_calls(
            [call(test_cube, c) for c in tscm_masks.values()], any_order=True
        )

    def test_get_scm_masks(self, test_cube):
        tsftlf_cube = "mocked 124"
        tland_mask_threshold = "mocked 51"

        land_mask = np.array(
            [
                [False, True, True, False],
                [False, True, False, True],
                [False, False, True, False],
            ]
        )
        test_cube._get_land_mask = MagicMock(return_value=land_mask)

        nh_mask = np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [True, True, True, True],
            ]
        )
        test_cube._get_nh_mask = MagicMock(return_value=nh_mask)

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

        result = test_cube._get_scm_masks(
            sftlf_cube=tsftlf_cube, land_mask_threshold=tland_mask_threshold
        )

        for label, array in expected.items():
            np.testing.assert_array_equal(array, result[label])
        test_cube._get_land_mask.assert_called_with(
            sftlf_cube=tsftlf_cube, land_mask_threshold=tland_mask_threshold
        )
        test_cube._get_nh_mask.assert_called_with()

    @pytest.mark.parametrize("transpose", [True, False])
    @pytest.mark.parametrize("input_format", ["scmcube", None])
    @pytest.mark.parametrize("sftlf_var", ["sftlf", "sftlf_other"])
    @pytest.mark.parametrize(
        "test_threshold",
        [(None), (0), (10), (30), (49), (49.9), (50), (50.1), (51), (60), (75), (100)],
    )
    def test_get_land_mask(
        self,
        test_cube,
        test_sftlf_cube,
        test_threshold,
        input_format,
        sftlf_var,
        transpose,
    ):
        test_cube.sftlf_var = sftlf_var
        original_data = test_sftlf_cube.cube.data

        if transpose:
            test_sftlf_cube.cube = iris.cube.Cube(
                data=np.transpose(test_sftlf_cube.cube.data)
            )
        test_cube.get_metadata_cube = MagicMock(return_value=test_sftlf_cube)

        test_land_fraction_input = (
            test_sftlf_cube if input_format is "scmcube" else None
        )

        if test_threshold is None:
            result = test_cube._get_land_mask(test_land_fraction_input)
            # test that default land fraction is 50%
            test_threshold = 50
        else:
            result = test_cube._get_land_mask(
                test_land_fraction_input, land_mask_threshold=test_threshold
            )

        # having got the result, we can now update test_land_fraction_input
        # for our assertions
        if test_land_fraction_input is None:
            test_cube.get_metadata_cube.assert_called_with(test_cube.sftlf_var)
            test_land_fraction_input = test_sftlf_cube

        # where it's land return False, otherwise True to match with masking
        # convention that True means masked
        expected = broadcast_to_shape(
            np.where(original_data > test_threshold, False, True),
            test_cube.cube.shape,
            [test_cube._lat_dim_number, test_cube._lon_dim_number],
        )
        np.testing.assert_array_equal(result, expected)

        if input_format is None:
            test_cube.get_metadata_cube.assert_called_with(sftlf_var)
        else:
            test_cube.get_metadata_cube.assert_not_called()

    @pytest.mark.parametrize("input", ["fail string", np.array([[1, 2], [3, 4]])])
    def test_get_land_mask_input_type_errors(self, test_cube, test_sftlf_cube, input):
        error_msg = re.escape(r"sftlf_cube must be an SCMCube instance")
        with pytest.raises(TypeError, match=error_msg):
            test_cube._get_land_mask(sftlf_cube=input)

    def test_get_land_mask_shape_errors(self, test_cube, test_sftlf_cube):
        error_msg = re.escape(
            r"the sftlf_cube data must be the same shape as the "
            r"cube's longitude-latitude grid"
        )

        wrong_shape_data = np.array([[1, 2], [3, 4]])
        test_sftlf_cube.cube = iris.cube.Cube(data=wrong_shape_data)

        with pytest.raises(AssertionError, match=error_msg):
            test_cube._get_land_mask(sftlf_cube=test_sftlf_cube)

        test_cube.get_metadata_cube = MagicMock(return_value=test_sftlf_cube)
        with pytest.raises(AssertionError, match=error_msg):
            test_cube._get_land_mask(sftlf_cube=None)

    def test_get_nh_mask(self, test_cube):
        result = test_cube._get_nh_mask()
        expected_base = np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [True, True, True, True],
            ]
        )
        expected = broadcast_to_shape(
            expected_base,
            test_cube.cube.shape,
            [test_cube._lat_dim_number, test_cube._lon_dim_number],
        )

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("transpose", [True, False])
    @pytest.mark.parametrize("input_format", ["scmcube", None])
    @pytest.mark.parametrize("areacella_var", ["areacella", "area_other"])
    def test_get_area_weights(
        self, test_cube, test_sftlf_cube, areacella_var, input_format, transpose
    ):
        test_cube.areacella_var = areacella_var

        expected = broadcast_to_shape(
            test_sftlf_cube.cube.data,
            test_cube.cube.shape,
            [test_cube._lat_dim_number, test_cube._lon_dim_number],
        )

        # we can use test_sftlf_cube here as all we need is an array of the
        # right shape
        if transpose:
            test_sftlf_cube.cube = iris.cube.Cube(
                data=np.transpose(test_sftlf_cube.cube.data)
            )
        test_cube.get_metadata_cube = MagicMock(return_value=test_sftlf_cube)

        test_areacella_input = test_sftlf_cube if input_format == "scmcube" else None

        result = test_cube._get_area_weights(areacella_scmcube=test_areacella_input)
        if input_format is None:
            test_cube.get_metadata_cube.assert_called_with(areacella_var)

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "areacella", ["not a cube", "cube attr not a cube", "iris_error", "misshaped"]
    )
    @pytest.mark.parametrize("areacella_var", ["areacella", "area_other"])
    def test_get_area_weights_workarounds(
        self, test_cube, test_sftlf_cube, areacella_var, areacella
    ):
        test_cube.areacella_var = areacella_var

        # can safely ignore these warnings here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*Using DEFAULT_SPHERICAL.*")
            expected = iris.analysis.cartography.area_weights(test_cube.cube)

        # we can use test_sftlf_cube here as all we need is an array of the
        # right shape
        if areacella == "iris_error":
            iris_error_msg = "no cube found"
            test_cube.get_metadata_cube = MagicMock(
                side_effect=ConstraintMismatchError(iris_error_msg)
            )
        elif areacella == "misshaped":
            misshaped_cube = SCMCube
            misshaped_cube.cube = iris.cube.Cube(data=np.array([1, 2]))
            test_cube.get_metadata_cube = MagicMock(return_value=misshaped_cube)
        elif areacella == "not a cube":
            test_cube.get_metadata_cube = MagicMock(return_value=areacella)
        elif areacella == "cube attr not a cube":
            weird_ob = MagicMock()
            weird_ob.cube = 23
            test_cube.get_metadata_cube = MagicMock(return_value=weird_ob)

        with pytest.warns(None) as record:
            result = test_cube._get_area_weights()
        test_cube.get_metadata_cube.assert_called_with(areacella_var)

        fallback_warn = re.escape(
            "Couldn't find/use areacella_cube, falling back to "
            "iris.analysis.cartography.area_weights"
        )
        radius_warn = re.escape("Using DEFAULT_SPHERICAL_EARTH_RADIUS.")
        if areacella == "iris_error":
            specific_warn = re.escape(iris_error_msg)
        elif areacella == "misshaped":
            specific_warn = re.escape(
                "the sftlf_cube data must be the same shape as the cube's "
                "longitude-latitude grid"
            )
        elif areacella == "not a cube":
            specific_warn = re.escape("'str' object has no attribute 'cube'")
        elif areacella == "cube attr not a cube":
            specific_warn = re.escape(
                "areacella cube which was found has cube attribute which isn't an iris cube"
            )

        assert len(record) == 3
        assert re.match(radius_warn, str(record[2].message))
        assert re.match(fallback_warn, str(record[1].message))
        assert re.match(specific_warn, str(record[0].message))

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
    ):
        expected_calendar = (
            test_cube.cube.coords("time")[0].units.calendar
            if out_calendar is None
            else out_calendar
        )

        tscm_timeseries_cubes = {"mocked 1": 198, "mocked 2": 248}

        tget_time_axis = np.array([1, 2, 3])
        mock_get_time_axis_in_calendar.return_value = tget_time_axis

        expected_idx = pd.Index(tget_time_axis, dtype="object", name="time")
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


class TestMarbleCMIP5Cube(TestSCMCube):
    tclass = MarbleCMIP5Cube
    troot_dir = TEST_DATA_MARBLE_CMIP5_DIR
    tactivity = "cmip5"
    texperiment = "1pctCO2"
    tmodeling_realm = "Amon"
    tvariable_name = "tas"
    tmodel = "CanESM2"
    tensemble_member = "r1i1p1"
    ttime_period = "185001-198912"
    tfile_ext = ".nc"

    def test_load_data_from_path(self, test_cube):
        tpath = "./somewhere/over/the/rainbow/test.nc"
        tids = {"id1": "mocked", "id2": 123}

        test_cube.get_load_data_from_identifiers_args_from_filepath = MagicMock(
            return_value=tids
        )
        test_cube.load_data_from_identifiers = MagicMock()

        test_cube.load_data_from_path(tpath)
        test_cube.get_load_data_from_identifiers_args_from_filepath.assert_called_with(
            tpath
        )
        test_cube.load_data_from_identifiers.assert_called_with(**tids)

    def test_get_filepath_from_load_data_from_identifiers_args(self, test_cube):
        tkwargs_list = [
            "root_dir",
            "activity",
            "experiment",
            "modeling_realm",
            "variable_name",
            "model",
            "ensemble_member",
            "time_period",
            "file_ext",
        ]
        tkwargs = {k: getattr(self, "t" + k) for k in tkwargs_list}

        mock_data_path = "here/there/everywhere"
        test_cube._get_data_directory = MagicMock(return_value=mock_data_path)

        mock_data_name = "here_there_file.nc"
        test_cube._get_data_filename = MagicMock(return_value=mock_data_name)

        for kwarg in tkwargs_list:
            with pytest.raises(AttributeError):
                getattr(test_cube, kwarg)

        result = test_cube.get_filepath_from_load_data_from_identifiers_args(**tkwargs)
        expected = join(mock_data_path, mock_data_name)

        assert result == expected

        for kwarg in tkwargs_list:
            assert getattr(test_cube, kwarg) == tkwargs[kwarg]

        assert test_cube._get_data_directory.call_count == 1
        assert test_cube._get_data_filename.call_count == 1

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
        [
            "1pctCO2/fx/sftlf/CanESM2/r0i0p0/sftlf_fx_CanESM2_1pctCO2_r0i0p0.nc",
            "cmip5/1pctCO2/fx/sftlf/CanESM2/r0i0p0/sftlf_fx_CanESM2_1pctCO2-r0i0p0.nc",
        ],
    )
    def test_get_load_data_from_identifiers_args_from_filepath_errors(
        self, test_cube, tpath
    ):
        error_msg = re.escape("Filepath does not look right: {}".format(tpath))
        with pytest.raises(ValueError, match=error_msg):
            test_cube.get_load_data_from_identifiers_args_from_filepath(tpath)

    def test_get_data_directory(self, test_cube):
        expected = join(
            self.troot_dir,
            self.tactivity,
            self.texperiment,
            self.tmodeling_realm,
            self.tvariable_name,
            self.tmodel,
            self.tensemble_member,
        )

        atts_to_set = [
            "root_dir",
            "activity",
            "experiment",
            "modeling_realm",
            "variable_name",
            "model",
            "ensemble_member",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        result = test_cube._get_data_directory()

        assert result == expected

    def test_get_data_filename(self, test_cube):
        expected = (
            "_".join(
                [
                    self.tvariable_name,
                    self.tmodeling_realm,
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
            "modeling_realm",
            "variable_name",
            "model",
            "ensemble_member",
            "time_period",
            "file_ext",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        result = test_cube._get_data_filename()

        assert result == expected

    def test_get_data_filename_no_time(self, test_cube):
        expected = (
            "_".join(
                [
                    self.tvariable_name,
                    self.tmodeling_realm,
                    self.tmodel,
                    self.texperiment,
                    self.tensemble_member,
                ]
            )
            + self.tfile_ext
        )

        atts_to_set = [
            "experiment",
            "modeling_realm",
            "variable_name",
            "model",
            "ensemble_member",
            "file_ext",
        ]
        for att in atts_to_set:
            setattr(test_cube, att, getattr(self, "t" + att))

        test_cube.time_period = None

        result = test_cube._get_data_filename()

        assert result == expected

    def test_get_variable_constraint_from_load_data_from_identifiers_args(
        self, test_cube
    ):
        tkwargs_list = [
            "root_dir",
            "activity",
            "experiment",
            "modeling_realm",
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
            cube_func=(lambda c: c.var_name == np.str(tkwargs_list["variable_name"]))
        )
        """

        result = test_cube.get_variable_constraint_from_load_data_from_identifiers_args(
            **tkwargs
        )
        assert isinstance(result, iris.Constraint)

    def test_get_metadata_load_arguments(self, test_cube):
        tmetadata_var = "mdata_var"
        atts_to_set = [
            "root_dir",
            "activity",
            "experiment",
            "modeling_realm",
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
            "modeling_realm": "fx",
            "variable_name": tmetadata_var,
            "model": test_cube.model,
            "ensemble_member": "r0i0p0",
            "time_period": None,
            "file_ext": test_cube.file_ext,
        }

        result = test_cube._get_metadata_load_arguments(tmetadata_var)

        assert result == expected
