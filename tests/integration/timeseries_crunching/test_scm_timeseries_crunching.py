"""
Test that crunching timeseries behaves as expected using files we have written

Helpful snippets:
python -c "from test_scm_timeseries_crunching import write_test_files, write_sftlf_file, write_area_file, write_data_file; wp = '.'; write_test_files(wp)"

from matplotlib import pyplot as plt
import iris.quickplot as qplt
qplt.pcolor(default_sftlf_cube); plt.gca().coastlines(); plt.show()  # maps
qplt.pcolor(cube); plt.gca().coastlines(); plt.show()  # maps

qplt.plot(cube[:, 1, 3]); plt.show()  # timeseries
"""
import datetime as dt
import os.path

import iris
import numpy as np
from conftest import assert_scmdata_frames_allclose
from openscm.scmdataframe import ScmDataFrame

from netcdf_scm.iris_cube_wrappers import MarbleCMIP5Cube

_root_dir = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(
    _root_dir,
    "cmip5",
    "experiment",
    "table",
    "rsdt",
    "model",
    "realisation",
    "rsdt_table_model_experiment_realisation_185001-185003.nc",
)
TEST_AREACEALLA_PATH = os.path.join(
    _root_dir,
    "cmip5",
    "experiment",
    "fx",
    "areacella",
    "model",
    "r0i0p0",
    "areacella_fx_model_experiment_r0i0p0.nc",
)
TEST_SFTLF_PATH = os.path.join(
    _root_dir,
    "cmip5",
    "experiment",
    "fx",
    "sftlf",
    "model",
    "r0i0p0",
    "sftlf_fx_model_experiment_r0i0p0.nc",
)


def test_scm_timeseries_crunching():
    tcube = MarbleCMIP5Cube()
    tcube.load_data_from_path(TEST_DATA_PATH)
    regions = [
        "World",
        "World|Land",
        "World|Ocean",
        "World|Northern Hemisphere",
        "World|Southern Hemisphere",
        "World|Northern Hemisphere|Land",
        "World|Southern Hemisphere|Land",
        "World|Northern Hemisphere|Ocean",
        "World|Southern Hemisphere|Ocean",
        "World|North Atlantic Ocean",
        "World|El Nino N3.4",
    ]

    time = [
        dt.datetime(1850, 1, 16, 12),
        dt.datetime(1850, 2, 15, 0),
        dt.datetime(1850, 3, 16, 12),
    ]

    # We do this by hand to make sure we haven't made an error. Yes, it is very slow
    # but that is the point.
    world_values = np.array(
        [
            (30 + 40 + 50 + 60) * 1.2
            + (110 + 120 + 190 + 260) * 2
            + (3 + 60 + 20 + 40) * 1.1,
            (0 + 15 + 45 + 90) * 1.2
            + (300 + 350 + 450 + 270) * 2
            + (10 + 70 + 90 + 130) * 1.1,
            (60 + 120 + 60 + 60) * 1.2
            + (510 + 432 + 220 + 280) * 2
            + (50 + 60 + 55 + 60) * 1.1,
        ]
    ) / (4 * 1.2 + 4 * 2 + 4 * 1.1)

    world_land_values = np.array(
        [
            (110 + 120) * 2 + (20) * 1.1,
            (300 + 350) * 2 + (90) * 1.1,
            (510 + 432) * 2 + (55) * 1.1,
        ]
    ) / (2 * 2 + 1 * 1.1)

    world_ocean_values = np.array(
        [
            (30 + 40 + 50 + 60) * 1.2 + (190 + 260) * 2 + (3 + 60 + 40) * 1.1,
            (0 + 15 + 45 + 90) * 1.2 + (450 + 270) * 2 + (10 + 70 + 130) * 1.1,
            (60 + 120 + 60 + 60) * 1.2 + (220 + 280) * 2 + (50 + 60 + 60) * 1.1,
        ]
    ) / (4 * 1.2 + 2 * 2 + 3 * 1.1)

    world_nh_values = np.array(
        [
            (30 + 40 + 50 + 60) * 1.2 + (110 + 120 + 190 + 260) * 2,
            (0 + 15 + 45 + 90) * 1.2 + (300 + 350 + 450 + 270) * 2,
            (60 + 120 + 60 + 60) * 1.2 + (510 + 432 + 220 + 280) * 2,
        ]
    ) / (4 * 1.2 + 4 * 2)

    world_sh_values = np.array(
        [
            (3 + 60 + 20 + 40) * 1.1,
            (10 + 70 + 90 + 130) * 1.1,
            (50 + 60 + 55 + 60) * 1.1,
        ]
    ) / (4 * 1.1)

    world_nh_land_values = np.array(
        [(110 + 120) * 2, (300 + 350) * 2, (510 + 432) * 2]
    ) / (2 * 2)

    world_sh_land_values = np.array([(20) * 1.1, (90) * 1.1, (55) * 1.1]) / (1 * 1.1)

    world_nh_ocean_values = np.array(
        [
            (30 + 40 + 50 + 60) * 1.2 + (190 + 260) * 2,
            (0 + 15 + 45 + 90) * 1.2 + (450 + 270) * 2,
            (60 + 120 + 60 + 60) * 1.2 + (220 + 280) * 2,
        ]
    ) / (4 * 1.2 + 2 * 2)

    world_sh_ocean_values = np.array(
        [(3 + 60 + 40) * 1.1, (10 + 70 + 130) * 1.1, (50 + 60 + 60) * 1.1]
    ) / (3 * 1.1)

    world_na_values = np.array([(260) * 2, (270) * 2, (280) * 2]) / (2)

    world_elnino_values = np.array([(190) * 2, (450) * 2, (220) * 2]) / (1 * 2)
    # get_area_mask(-5, -170, 5, -120)

    data = np.vstack(
        [
            world_values,
            world_land_values,
            world_ocean_values,
            world_nh_values,
            world_sh_values,
            world_nh_land_values,
            world_sh_land_values,
            world_nh_ocean_values,
            world_sh_ocean_values,
            world_na_values,
            world_elnino_values,
        ]
    ).T

    exp = ScmDataFrame(
        data=data,
        index=time,
        columns={
            "model": "unspecified",
            "scenario": "experiment",
            "region": regions,
            "variable": "rsdt",
            "unit": "W m^-2",
            "climate_model": "model",
            "activity_id": "cmip5",
            "member_id": "realisation",
            "variable_standard_name": "toa_incoming_shortwave_flux",
            "mip_era": "CMIP5",
        },
    )
    exp.metadata = {"calendar": "gregorian"}
    res = tcube.get_scm_timeseries(masks=regions)
    assert_scmdata_frames_allclose(res, exp)


def write_test_files(write_path):
    lat = iris.coords.DimCoord(
        np.array([70, 5, -45]),
        bounds=np.array([[30, 90], [0, 30], [-90, 0]]),
        standard_name="latitude",
        units="degrees",
    )
    lon = iris.coords.DimCoord(
        np.array([45, 135, 225, 315]),
        bounds=np.array([[0, 90], [90, 180], [180, 270], [270, 360]]),
        standard_name="longitude",
        units="degrees",
        circular=True,
    )
    write_sftlf_file(write_path, lat, lon)
    write_area_file(write_path, lat, lon)
    write_data_file(write_path, lat, lon)


def write_sftlf_file(write_path, lat, lon):
    data = np.array([[0, 30, 0, 10], [80, 100, 0, 50], [20, 10, 51, 15]])
    cube = iris.cube.Cube(
        data,
        standard_name="land_area_fraction",
        var_name="sftlf",
        units="%",
        dim_coords_and_dims=[(lat, 0), (lon, 1)],
    )
    iris.save(cube, TEST_SFTLF_PATH)


def write_area_file(write_path, lat, lon):
    data = np.array([[1.2, 1.2, 1.2, 1.2], [2, 2, 2, 2], [1.1, 1.1, 1.1, 1.1]])
    cube = iris.cube.Cube(
        data,
        standard_name="cell_area",
        var_name="areacella",
        units="m^2",
        dim_coords_and_dims=[(lat, 0), (lon, 1)],
    )
    iris.save(cube, TEST_AREACEALLA_PATH)


def write_data_file(write_path, lat, lon):
    time = iris.coords.DimCoord(
        np.array([15.5, 45, 74.5]),
        standard_name="time",
        units="days since 1850-01-01 00:00:00",
    )
    time.guess_bounds()

    data = np.array(
        [
            [[30, 40, 50, 60], [110, 120, 190, 260], [3, 60, 20, 40]],
            [[0, 15, 45, 90], [300, 350, 450, 270], [10, 70, 90, 130]],
            [[60, 120, 60, 60], [510, 432, 220, 280], [50, 60, 55, 60]],
        ]
    )
    cube = iris.cube.Cube(
        data,
        standard_name="toa_incoming_shortwave_flux",
        var_name="rsdt",
        units="W m-2",
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )
    iris.save(cube, TEST_DATA_PATH)
