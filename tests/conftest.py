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

TEST_DATA_KNMI_DIR = join(TEST_DATA_ROOT_DIR, "knmi-climate-explorer")

TEST_DATA_MARBLE_CMIP5_DIR = join(TEST_DATA_ROOT_DIR, "marble-cmip5")
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
TEST_SFTLF_FILE = join(
    TEST_DATA_MARBLE_CMIP5_DIR,
    "cmip5",
    "1pctCO2",
    "fx",
    "sftlf",
    "CanESM2",
    "r0i0p0",
    "sftlf_fx_CanESM2_1pctCO2_r0i0p0.nc",
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
TEST_ACCESS_CMIP5_FILE = join(
    TEST_DATA_MARBLE_CMIP5_DIR,
    "cmip5",
    "rcp45",
    "Amon",
    "tas",
    "ACCESS1-0",
    "r1i1p1",
    "tas_Amon_ACCESS1-0_rcp45_r1i1p1_200601-204912.nc",
)

TEST_DATA_CMIP6Input4MIPS_DIR = join(TEST_DATA_ROOT_DIR, "cmip6input4mips")
TEST_CMIP6_HISTORICAL_CONCS_FILE = join(
    TEST_DATA_CMIP6Input4MIPS_DIR,
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

TEST_DATA_CMIP6Output_DIR = join(TEST_DATA_ROOT_DIR, "cmip6output")
TEST_CMIP6_OUTPUT_FILE = join(
    TEST_DATA_CMIP6Output_DIR,
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
TEST_CMIP6_OUTPUT_FILE_MISSING_BOUNDS = join(
    TEST_DATA_CMIP6Output_DIR,
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
TEST_CMIP6_OUTPUT_FILE_1_UNIT = join(
    TEST_DATA_CMIP6Output_DIR,
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

TEST_DATA_CMIP6_CRUNCH_OUTPUT = join(
    TEST_DATA_ROOT_DIR, "expected-crunching-output", "cmip6output"
)
TEST_DATA_MARBLE_CMIP5_CRUNCH_OUTPUT = join(
    TEST_DATA_ROOT_DIR, "expected-crunching-output", "marble-cmip5"
)

TEST_DATA_NETCDFSCM_NCS_DIR = join(TEST_DATA_ROOT_DIR, "netcdf-scm-ncs")
TEST_DATA_NETCDFSCM_NC_FILE = join(
    TEST_DATA_ROOT_DIR, "netcdf-scm_tas_Amon_ACCESS1-0_rcp45_r1i1p1_200601-204912.nc"
)

tdata_required = pytest.mark.skipif(
    not isdir(TEST_DATA_ROOT_DIR), reason="test data required"
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
def test_generic_tas_cube():
    test_generic_tas_cube = SCMCube()
    # can safely ignore these warnings here
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Missing CF-netCDF measure variable 'areacella'"
        )
        test_generic_tas_cube.cube = iris.load_cube(TEST_TAS_FILE)

    return test_generic_tas_cube


def run_crunching_comparison(res, expected, update=False):
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
                        res_df = load_scmdataframe(res_f).timeseries().sort_index()
                        exp_df = load_scmdataframe(exp_f).timeseries().sort_index()
                        pd.testing.assert_frame_equal(res_df, exp_df, check_like=True)

    if update:
        print("Updated {}".format(expected))
        pytest.skip()
