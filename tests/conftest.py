from os.path import join, dirname, abspath, isdir
import warnings


import pytest
import numpy as np
import iris
import cf_units as unit


from netcdf_scm.iris_cube_wrappers import SCMCube


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


tdata_required = pytest.mark.skipif(
    not isdir(TEST_DATA_ROOT_DIR), reason="test data required"
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
