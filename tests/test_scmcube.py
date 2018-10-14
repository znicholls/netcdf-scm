from os.path import join, isdir, isfile, splitext, dirname, abspath
from unittest.mock import patch, MagicMock
import warnings


import pytest
from pytest import raises
import numpy as np
import re
import iris
from iris.exceptions import ConstraintMismatchError
from iris.analysis.cartography import DEFAULT_SPHERICAL_EARTH_RADIUS
from iris.util import broadcast_to_shape
import cf_units
import cf_units as unit
from pymagicc.io import MAGICCData
import pandas as pd
from pandas.testing import assert_frame_equal


from netcdf_scm.iris_cube_wrappers import SCMCube, MarbleCMIP5Cube


TEST_DATA_ROOT_DIR = join(dirname(abspath(__file__)), "test_data")
TEST_DATA_MARBLE_CMIP5_DIR = join(TEST_DATA_ROOT_DIR, "marble_cmip5")
TEST_TAS_FILE = join(
    TEST_DATA_MARBLE_CMIP5_DIR,
    "cmip5",
    "1pctCO2",
    "Amon",
    "tas",
    "CanESM2",
    "r1i1p1",
    "tas_Amon_CanESM2_1pctCO2_r1i1p1_185001-198912.nc",
)
TEST_AREACELLA_FILE = join(
    TEST_DATA_MARBLE_CMIP5_DIR,
    "cmip5",
    "1pctCO2",
    "fx",
    "areacella",
    "CanESM2",
    "r0i0p0",
    "areacella_fx_CanESM2_1pctCO2_r0i0p0.nc",
)


def get_test_cube_lon():
    lon = iris.coords.DimCoord(
        np.array([45, 135, 225, 315]),
        standard_name="longitude",
        units=unit.Unit("degrees"),
        long_name="longitude",
        var_name="lon",
        circular=True,
    )
    lon.guess_bounds()
    return lon


def get_test_cube_lat():
    lat = iris.coords.DimCoord(
        np.array([60, 0, -60]),
        standard_name="latitude",
        units=unit.Unit("degrees"),
        long_name="latitude",
        var_name="lat",
    )
    lat.guess_bounds()
    return lat


def get_test_cube_time():
    return iris.coords.DimCoord(
        np.array([365, 365 * 2, 365 * 3, 365 * 3 + 180]),
        standard_name="time",
        units=unit.Unit("days since 1850-1-1", calendar="365_day"),
        long_name="time",
        var_name="time",
    )


def get_test_cube_attributes():
    return {
        "Creator": "Blinky Bill",
        "Supervisor": "Patch",
        "attribute 3": "attribute 3",
        "attribute d": "hello, attribute d",
    }


@pytest.fixture(scope="function")
def test_cube(request):
    test_cube = request.cls.tclass()

    test_data = np.ma.masked_array(
        [
            [[0, 0.5, 1, 3], [0.0, 0.15, 0.25, 0.3], [-4, -5, -6, -7]],
            [[9, 9, 7, 6], [0, 1, 2, 3], [5, 4, 3, 2]],
            [[10, 14, 12, 13], [0, 1, 2, 3], [4.1, 5.2, 6.2, 7.3]],
            [[10, 18, 12, 13], [0, 1, 2, 3], [4.1, 5.2, 6.2, 7.3]],
        ],
        mask=False,
    )

    test_cube.cube = iris.cube.Cube(
        test_data,
        standard_name="air_temperature",
        long_name="air_temperature",
        var_name="air_temperature",
        dim_coords_and_dims=[
            (get_test_cube_time(), 0),
            (get_test_cube_lat(), 1),
            (get_test_cube_lon(), 2),
        ],
        units=unit.Unit("degC"),
        attributes=get_test_cube_attributes(),
    )

    return test_cube


@pytest.fixture(scope="function")
def test_sftlf_cube(request):
    test_sftlf_cube = request.cls.tclass()

    test_data = np.ma.masked_array(
        [[90, 49.9, 50.0, 50.1], [100, 49, 50, 51], [51, 30, 10, 0]], mask=False
    )
    test_sftlf_cube.cube = iris.cube.Cube(test_data)
    test_sftlf_cube.cube.standard_name = "land_area_fraction"

    test_sftlf_cube.cube.add_dim_coord(get_test_cube_lat(), 0)
    test_sftlf_cube.cube.add_dim_coord(get_test_cube_lon(), 1)

    return test_sftlf_cube


class TestSCMCube(object):
    tclass = SCMCube

    def run_test_of_method_to_overload(
        self, test_cube, method_to_overload, junk_args={}
    ):
        if type(test_cube) is SCMCube:
            with pytest.raises(NotImplementedError):
                getattr(test_cube, method_to_overload)(**junk_args)
        else:
            assert False, "Overload {} in your subclass".format(method_to_overload)

    @patch("netcdf_scm.iris_cube_wrappers.iris.load_cube")
    def test_load_data(self, mock_iris_load_cube, test_cube):
        tfile = "hello_world_test.nc"
        test_cube.get_file_from_load_data_args = MagicMock(return_value=tfile)

        vcons = 12.195
        test_cube.get_variable_constraint_from_load_data_args = MagicMock(
            return_value=vcons
        )

        test_cube.get_metadata_cube = MagicMock()

        lcube_return = 9848
        mock_iris_load_cube.return_value = lcube_return

        tkwargs = {
            "variable_name": "fco2antt",
            "modeling_realm": "Amon",
            "model": "CanESM2",
            "experiment": "1pctCO2",
        }
        test_cube.load_data(**tkwargs)

        test_cube.get_file_from_load_data_args.assert_called_with(**tkwargs)
        test_cube.get_variable_constraint_from_load_data_args.assert_called_with(
            **tkwargs
        )
        mock_iris_load_cube.assert_called_with(tfile, vcons)
        test_cube.get_metadata_cube.assert_not_called()
        assert test_cube.cube == lcube_return

    # this is really an integration test, maybe should be moved/split...
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

    def test_load_missing_variable_error(self, test_cube):
        tfile = TEST_TAS_FILE
        test_cube.get_file_from_load_data_args = MagicMock(return_value=tfile)

        bad_constraint = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str("misnamed_var"))
        )
        test_cube.get_variable_constraint_from_load_data_args = MagicMock(
            return_value=bad_constraint
        )

        with raises(ConstraintMismatchError, match="no cubes found"):
            test_cube.load_data(mocked_out="mocked")

    def test_get_file_from_load_data_args(self, test_cube):
        self.run_test_of_method_to_overload(test_cube, "get_file_from_load_data_args")

    def test_get_variable_constraint_from_load_data_args(self, test_cube):
        self.run_test_of_method_to_overload(
            test_cube, "get_variable_constraint_from_load_data_args"
        )

    def test_get_data_path(self, test_cube):
        self.run_test_of_method_to_overload(test_cube, "_get_data_path")

    def test_get_data_name(self, test_cube):
        self.run_test_of_method_to_overload(test_cube, "_get_data_name")

    def test_get_metadata_load_arguments(self, test_cube):
        self.run_test_of_method_to_overload(
            test_cube,
            "_get_metadata_load_arguments",
            junk_args={"metadata_variable": "mdata_var"},
        )

    @patch.object(tclass, "load_data")
    def test_get_metadata_cube(self, mock_load_data, test_cube):
        tvar = "tmdata_var"
        tload_arg_dict = {"Arg 1": 12, "Arg 2": "Val 2"}

        test_cube._get_metadata_load_arguments = MagicMock(return_value=tload_arg_dict)

        result = test_cube.get_metadata_cube(tvar)

        assert type(result) == type(test_cube)

        test_cube._get_metadata_load_arguments.assert_called_with(tvar)
        mock_load_data.assert_called_with(**tload_arg_dict)

    def test_get_scm_timeseries(self, test_sftlf_cube, test_cube):
        tsftlf_cube = "mocked 124"
        tland_mask_threshold = "mocked 51"
        tareacella_scmcube = "mocked 4389"

        test_cubes_return = {"NH_OCEAN": 4, "SH_LAND": 12}
        test_cube.get_scm_timeseries_cubes = MagicMock(return_value=test_cubes_return)

        test_conversion_return = pd.DataFrame(data=np.array([1, 2, 3]))
        test_cube._convert_scm_timeseries_cubes_to_OpenSCMData = MagicMock(
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
        test_cube._convert_scm_timeseries_cubes_to_OpenSCMData.assert_called_with(
            test_cubes_return
        )

        assert_frame_equal(result, test_conversion_return)

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

        lon_dim = test_cube.cube.coord_dims("longitude")[0]
        lat_dim = test_cube.cube.coord_dims("latitude")[0]
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
        nh_mask = np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [True, True, True, True],
            ]
        )

        test_cube._get_land_mask = MagicMock(return_value=land_mask)
        test_cube._get_nh_mask = MagicMock(return_value=nh_mask)

        nh_land_mask = np.array(
            [
                [False, True, True, False],
                [False, True, False, True],
                [True, True, True, True],
            ]
        )
        # check our logic while we're here
        np.testing.assert_array_equal(nh_land_mask, np.logical_or(nh_mask, land_mask))

        expected = {
            "GLOBAL": np.full(nh_mask.shape, False),
            "NH_LAND": nh_land_mask,
            "SH_LAND": np.logical_or(~nh_mask, land_mask),
            "NH_OCEAN": np.logical_or(nh_mask, ~land_mask),
            "SH_OCEAN": np.logical_or(~nh_mask, ~land_mask),
        }

        result = test_cube._get_scm_masks(
            sftlf_cube=tsftlf_cube, land_mask_threshold=tland_mask_threshold
        )

        for label, array in result.items():
            np.testing.assert_array_equal(array, expected[label])
        test_cube._get_land_mask.assert_called_with(
            sftlf_cube=tsftlf_cube, land_mask_threshold=tland_mask_threshold
        )
        test_cube._get_nh_mask.assert_called_with()

    @pytest.mark.parametrize("transpose", [True, False])
    @pytest.mark.parametrize("input_format", ["nparray", "scmcube", None])
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
        test_cube._sftlf_var = sftlf_var
        test_cube.get_metadata_cube = MagicMock(return_value=test_sftlf_cube)

        if input_format is "nparray":
            test_land_fraction_input = test_sftlf_cube.cube.data
            if transpose:
                test_land_fraction_input = np.transpose(test_land_fraction_input)
        elif input_format is "scmcube":
            test_land_fraction_input = test_sftlf_cube
        else:
            test_land_fraction_input = None

        if test_threshold is None:
            result = test_cube._get_land_mask(test_land_fraction_input)
            # default land fraction is 50%
            test_threshold = 50
        else:
            result = test_cube._get_land_mask(
                test_land_fraction_input, land_mask_threshold=test_threshold
            )

        # having got the result, we can now update test_land_fraction_input
        # for our assertions
        if test_land_fraction_input is None:
            test_cube.get_metadata_cube.assert_called_with(test_cube._sftlf_var)
            test_land_fraction_input = test_sftlf_cube

        # where it's land return False, otherwise True to match with masking
        # convention that True means masked
        expected = broadcast_to_shape(
            np.where(test_sftlf_cube.cube.data > test_threshold, False, True),
            test_cube.cube.shape,
            [test_cube._lat_dim_number, test_cube._lon_dim_number],
        )

        np.testing.assert_array_equal(result, expected)
        if input_format is None:
            test_cube.get_metadata_cube.assert_called_with(sftlf_var)
        else:
            test_cube.get_metadata_cube.assert_not_called()

    def test_get_land_mask_errors(self, test_cube, test_sftlf_cube):
        error_msg = re.escape(
            r"sftlf_cube must be a numpy.ndarray if it's not an SCMCube instance"
        )
        with pytest.raises(AssertionError, match=error_msg):
            test_cube._get_land_mask(sftlf_cube="fail string")

        wrong_shape_data = np.array([[1, 2], [3, 4]])
        error_msg = re.escape(
            r"the sftlf_cube data must be the same shape as (or the transpose of) the "
            r"cube's longitude-latitude grid"
        )
        with pytest.raises(AssertionError, match=error_msg):
            test_cube._get_land_mask(sftlf_cube=wrong_shape_data)

        test_sftlf_cube.cube = iris.cube.Cube(data=wrong_shape_data)
        test_cube.get_metadata_cube = MagicMock(return_value=test_sftlf_cube)
        with pytest.raises(AssertionError, match=error_msg):
            test_cube._get_land_mask(sftlf_cube=wrong_shape_data)

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
    @pytest.mark.parametrize(
        "valid_areacella", ["valid", "not a cube", "iris_error", "misshaped"]
    )
    @pytest.mark.parametrize("areacella_var", ["areacella", "area_other"])
    def test_get_area_weights(
        self,
        test_cube,
        test_sftlf_cube,
        areacella_var,
        valid_areacella,
        input_format,
        transpose,
    ):
        test_cube._areacella_var = areacella_var

        if valid_areacella == "valid":
            expected = broadcast_to_shape(
                test_sftlf_cube.cube.data,
                test_cube.cube.shape,
                [test_cube._lat_dim_number, test_cube._lon_dim_number],
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*Using DEFAULT_SPHERICAL.*")
                expected = iris.analysis.cartography.area_weights(test_cube.cube)

        # we can use test_sftlf_cube here as all we need is an array of the
        # right shape
        if valid_areacella == "iris_error":
            iris_error_msg = "no cube found"
            test_cube.get_metadata_cube = MagicMock(
                side_effect=ConstraintMismatchError(iris_error_msg)
            )
        elif valid_areacella == "misshaped":
            misshaped_cube = SCMCube
            misshaped_cube.cube = iris.cube.Cube(data=np.array([1, 2]))
            test_cube.get_metadata_cube = MagicMock(return_value=misshaped_cube)
        elif valid_areacella == "valid":
            if transpose:
                test_sftlf_cube.cube = iris.cube.Cube(
                    data=np.transpose(test_sftlf_cube.cube.data)
                )
            test_cube.get_metadata_cube = MagicMock(return_value=test_sftlf_cube)
        else:
            test_cube.get_metadata_cube = MagicMock(return_value=valid_areacella)

        if input_format == "scmcube":
            test_areacella_input = test_sftlf_cube
        else:
            test_areacella_input = None

        if valid_areacella == "valid":
            result = test_cube._get_area_weights(areacella_scmcube=test_areacella_input)
            if input_format is None:
                test_cube.get_metadata_cube.assert_called_with(areacella_var)
        else:
            with pytest.warns(None) as record:
                result = test_cube._get_area_weights()
            test_cube.get_metadata_cube.assert_called_with(areacella_var)

            fallback_warn = re.escape(
                "Couldn't find/use areacella_cube, falling back to "
                "iris.analysis.cartography.area_weights"
            )
            radius_warn = re.escape("Using DEFAULT_SPHERICAL_EARTH_RADIUS.")
            if valid_areacella == "iris_error":
                specific_warn = re.escape(iris_error_msg)
            elif valid_areacella == "misshaped":
                specific_warn = re.escape(
                    "the sftlf_cube data must be the same shape as (or the "
                    "transpose of) the cube's longitude-latitude grid"
                )
            else:
                specific_warn = re.escape("'str' object has no attribute 'cube'")

            assert len(record) == 3
            assert re.match(radius_warn, str(record[2].message))
            assert re.match(fallback_warn, str(record[1].message))
            assert re.match(specific_warn, str(record[0].message))

        np.testing.assert_array_equal(result, expected)

    def test_convert_scm_timeseries_cubes_to_OpenSCMData(self, test_cube):
        # TODO: test switches and errors
        expected_calendar = test_cube.cube.coords("time")[0].units.calendar

        global_cube = type(test_cube)()
        global_cube.cube = test_cube.cube.copy()
        global_cube.cube.data = 2 * global_cube.cube.data
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

        result = test_cube._convert_scm_timeseries_cubes_to_OpenSCMData(
            test_timeseries_cubes
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
        pd.testing.assert_frame_equal(result.df, expected.df)


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

    def test_get_file_from_load_data_args(self, test_cube):
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
        test_cube._get_data_path = MagicMock(return_value=mock_data_path)

        mock_data_name = "here_there_file.nc"
        test_cube._get_data_name = MagicMock(return_value=mock_data_name)

        for kwarg in tkwargs_list:
            with pytest.raises(AttributeError):
                getattr(test_cube, kwarg)

        result = test_cube.get_file_from_load_data_args(**tkwargs)
        expected = join(mock_data_path, mock_data_name)

        assert result == expected

        for kwarg in tkwargs_list:
            assert getattr(test_cube, kwarg) == tkwargs[kwarg]

        assert test_cube._get_data_path.call_count == 1
        assert test_cube._get_data_name.call_count == 1

    def test_get_data_path(self, test_cube):
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

        result = test_cube._get_data_path()

        assert result == expected

    def test_get_data_name(self, test_cube):
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

        result = test_cube._get_data_name()

        assert result == expected

    def test_get_variable_constraint_from_load_data_args(self, test_cube):
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

        result = test_cube.get_variable_constraint_from_load_data_args(**tkwargs)
        assert isinstance(result, iris.Constraint)

        # impossible to do other tests as far as I can tell because you have to pass a
        # local function in both the test and the argument, help welcome. expected is
        # here.
        expected = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str(tkwargs_list["variable_name"]))
        )

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
