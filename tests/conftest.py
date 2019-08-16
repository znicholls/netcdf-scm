import shutil
import warnings
from os import makedirs, path, walk
from os.path import abspath, dirname, isdir, join

import cf_units as unit
import iris
import numpy as np
import pandas as pd
import pytest

from netcdf_scm.io import load_scmdataframe
from netcdf_scm.iris_cube_wrappers import (
    CMIP6Input4MIPsCube,
    CMIP6OutputCube,
    MarbleCMIP5Cube,
    SCMCube,
)

TEST_DATA_ROOT_DIR = join(dirname(abspath(__file__)), "test-data")


@pytest.fixture
def test_data_root_dir():
    if not isdir(TEST_DATA_ROOT_DIR):
        pytest.skip("test data required")
    return TEST_DATA_ROOT_DIR


@pytest.fixture
def test_data_knmi_dir(test_data_root_dir):
    return join(test_data_root_dir, "knmi-climate-explorer")


@pytest.fixture
def test_data_marble_cmip5_dir(test_data_root_dir):
    return join(test_data_root_dir, "marble-cmip5")


@pytest.fixture
def test_tas_file(test_data_marble_cmip5_dir):
    return join(
        test_data_marble_cmip5_dir,
        "cmip5",
        "1pctCO2",
        "Amon",
        "tas",
        "CanESM2",
        "r1i1p1",
        "tas_Amon_CanESM2_1pctCO2_r1i1p1_185001-198912.nc",
    )


@pytest.fixture
def test_sftlf_file(test_data_marble_cmip5_dir):
    return join(
        test_data_marble_cmip5_dir,
        "cmip5",
        "1pctCO2",
        "fx",
        "sftlf",
        "CanESM2",
        "r0i0p0",
        "sftlf_fx_CanESM2_1pctCO2_r0i0p0.nc",
    )


@pytest.fixture
def test_areacella_file(test_data_marble_cmip5_dir):
    return join(
        test_data_marble_cmip5_dir,
        "cmip5",
        "1pctCO2",
        "fx",
        "areacella",
        "CanESM2",
        "r0i0p0",
        "areacella_fx_CanESM2_1pctCO2_r0i0p0.nc",
    )


@pytest.fixture
def test_access_cmip5_file(test_data_marble_cmip5_dir):
    return join(
        test_data_marble_cmip5_dir,
        "cmip5",
        "rcp45",
        "Amon",
        "tas",
        "ACCESS1-0",
        "r1i1p1",
        "tas_Amon_ACCESS1-0_rcp45_r1i1p1_200601-204912.nc",
    )


@pytest.fixture
def test_data_cmip6input4mips_dir(test_data_root_dir):
    return join(test_data_root_dir, "cmip6input4mips")


@pytest.fixture
def test_cmip6input4mips_historical_concs_file(test_data_cmip6input4mips_dir):
    return join(
        test_data_cmip6input4mips_dir,
        "input4MIPs",
        "CMIP6",
        "CMIP",
        "UoM",
        "UoM-CMIP-1-2-0",
        "atmos",
        "yr",
        "mole-fraction-of-so2f2-in-air",
        "gr1-GMNHSH",
        "v20100304",
        "mole-fraction-of-so2f2-in-air_input4MIPs_GHGConcentrations_CMIP_UoM-CMIP-1-2-0_gr1-GMNHSH_0000-2014.nc",
    )


@pytest.fixture
def test_cmip6input4mips_projection_concs_file(test_data_cmip6input4mips_dir):
    return join(
        test_data_cmip6input4mips_dir,
        "input4MIPs",
        "CMIP6",
        "ScenarioMIP",
        "UoM",
        "UoM-MESSAGE-GLOBIOM-ssp245-1-2-1",
        "atmos",
        "mon",
        "mole-fraction-of-carbon-dioxide-in-air",
        "gn-15x360deg",
        "v20100304",
        "mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-MESSAGE-GLOBIOM-ssp245-1-2-1_gn-15x360deg_201501-203012.nc",
    )


@pytest.fixture
def test_data_cmip6output_dir(test_data_root_dir):
    return join(test_data_root_dir, "cmip6output")


@pytest.fixture
def test_cmip6_output_file(test_data_cmip6output_dir):
    return join(
        test_data_cmip6output_dir,
        "CMIP6",
        "CMIP",
        "BCC",
        "BCC-CSM2-MR",
        "1pctCO2",
        "r1i1p1f1",
        "Amon",
        "rlut",
        "gn",
        "v20181015",
        "rlut_Amon_BCC-CSM2-MR_1pctCO2_r1i1p1f1_gn_185001-185912.nc",
    )


@pytest.fixture
def test_cmip6_output_file_missing_bounds(test_data_cmip6output_dir):
    return join(
        test_data_cmip6output_dir,
        "CMIP6",
        "ScenarioMIP",
        "IPSL",
        "IPSL-CM6A-LR",
        "ssp126",
        "r1i1p1f1",
        "Lmon",
        "cSoilFast",
        "gr",
        "v20190121",
        "cSoilFast_Lmon_IPSL-CM6A-LR_ssp126_r1i1p1f1_gr_201501-210012.nc",
    )


@pytest.fixture
def test_cmip6_output_file_1_unit(test_data_cmip6output_dir):
    return join(
        test_data_cmip6output_dir,
        "CMIP6",
        "CMIP",
        "CNRM-CERFACS",
        "CNRM-CM6-1",
        "historical",
        "r1i1p1f2",
        "Lmon",
        "lai",
        "gr",
        "v20180917",
        "lai_Lmon_CNRM-CM6-1_historical_r1i1p1f2_gr_200001-201412.nc",
    )


# land file (lai?)
# hfds file: gn and gr
# thetao file: gn and gr (eventually...)
#     - add_volcello test (from tos-hfds branch)


@pytest.fixture
def test_cmip6_crunch_output(test_data_root_dir):
    return join(test_data_root_dir, "expected-crunching-output", "cmip6output", "CMIP6")


@pytest.fixture
def test_marble_cmip5_crunch_output(test_data_root_dir):
    return join(
        test_data_root_dir, "expected-crunching-output", "marble-cmip5", "cmip5"
    )


@pytest.fixture
def test_data_netcdfscm_ncs_dir(test_data_root_dir):
    return join(test_data_root_dir, "netcdf-scm-ncs")


@pytest.fixture
def test_data_netcdfscm_nc_file(test_data_root_dir):
    return join(
        test_data_root_dir,
        "netcdf-scm_tas_Amon_ACCESS1-0_rcp45_r1i1p1_200601-204912.nc",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--update-expected-files",
        action="store_true",
        default=False,
        help="Overwrite expected files",
    )


@pytest.fixture
def update_expected_files(request):
    return request.config.getoption("--update-expected-files")


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


def create_cube(cube_cls):
    c = cube_cls()
    test_data = np.ma.masked_array(
        [
            [[0, 0.5, 1, 3], [0.0, 0.15, 0.25, 0.3], [-4, -5, -6, -7]],
            [[9, 9, 7, 6], [0, 1, 2, 3], [5, 4, 3, 2]],
            [[10, 14, 12, 13], [0, 1, 2, 3], [4.1, 5.2, 6.2, 7.3]],
            [[10, 18, 12, 13], [0, 1, 2, 3], [4.1, 5.2, 6.2, 7.3]],
        ],
        mask=False,
    )

    c.cube = iris.cube.Cube(
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

    return c


@pytest.fixture(scope="function")
def test_cube(request):
    return create_cube(request.cls.tclass)


@pytest.fixture(
    scope="function",
    params=[SCMCube, MarbleCMIP5Cube, CMIP6Input4MIPsCube, CMIP6OutputCube],
)
def test_all_cubes(request):
    return create_cube(request.param)


def create_sftlf_cube(cube_cls):
    c = cube_cls()
    c._loaded_paths.append("test_sftlf.nc")

    test_data = np.ma.masked_array(
        [[90, 49.9, 50.0, 50.1], [100, 49, 50, 51], [51, 30, 10, 0]], mask=False
    )
    c.cube = iris.cube.Cube(test_data)
    c.cube.standard_name = "land_area_fraction"

    c.cube.add_dim_coord(get_test_cube_lat(), 0)
    c.cube.add_dim_coord(get_test_cube_lon(), 1)

    return c


@pytest.fixture(scope="function")
def test_sftlf_cube(request):
    return create_sftlf_cube(request.cls.tclass)


@pytest.fixture(scope="function")
def test_generic_tas_cube(test_tas_file):
    test_generic_tas_cube = SCMCube()
    # can safely ignore these warnings here
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Missing CF-netCDF measure variable 'areacella'"
        )
        test_generic_tas_cube.cube = iris.load_cube(test_tas_file)

    return test_generic_tas_cube


@pytest.fixture
def run_crunching_comparison(assert_scmdata_frames_allclose):
    def _do_comparison(res, expected, update=False):
        """Run test that crunched files are unchanged

        Parameters
        ----------
        res : str
            Directory written as part of the test
        expected : str
            Directory against which the comparison should be done
        update : bool
            If True, don't perform the test and instead simply
            overwrite the ``expected`` with ``res``

        Raises
        ------
        AssertionError
            If ``update`` is ``False`` and ``res`` and ``expected``
            are not identical.
        """
        paths_to_walk = [expected, res] if not update else [res]
        for p in paths_to_walk:
            for dirpath, _, filenames in walk(p):
                if filenames:
                    if update:
                        path_to_check = dirpath.replace(res, expected)
                        if not path.exists(path_to_check):
                            makedirs(path_to_check)

                    for f in filenames:
                        res_f = join(dirpath, f)
                        exp_f = res_f.replace(res, expected)
                        if update:
                            shutil.copy(res_f, exp_f)
                        else:
                            res_scmdf = load_scmdataframe(res_f)
                            exp_scmdf = load_scmdataframe(exp_f)
                            assert_scmdata_frames_allclose(res_scmdf, exp_scmdf)

        if update:
            print("Updated {}".format(expected))
            pytest.skip()

    return _do_comparison


@pytest.fixture
def assert_scmdata_frames_allclose():
    def _do_assertion(res_scmdf, exp_scmdf):
        res_df = res_scmdf.timeseries().sort_index()
        assert (
            (res_df.values > -10 ** 5) & (res_df.values < 10 ** 5)
        ).all(), "Failed sanity check"

        exp_df = exp_scmdf.timeseries().sort_index()
        pd.testing.assert_frame_equal(res_df, exp_df, check_like=True)
        for base, check in [(exp_scmdf, res_scmdf), (res_scmdf, exp_scmdf)]:
            for k, v in base.metadata.items():
                if k == "crunch_netcdf_scm_version":
                    continue  # will change with version
                if isinstance(v, (np.ndarray, np.float, np.int)):
                    np.testing.assert_allclose(v, check.metadata[k])
                else:
                    v == check.metadata[k]

    return _do_assertion
