"""
Test that crunching timeseries behaves as expected using files we have written

Helpful snippets:
python -c "from test_scm_timeseries_crunching import write_test_files; wp = '.'; write_test_files(wp)"

from matplotlib import pyplot as plt
import iris.quickplot as qplt
qplt.pcolor(default_sftlf_cube); plt.gca().coastlines(); plt.show()  # maps
qplt.pcolor(cube); plt.gca().coastlines(); plt.show()  # maps

qplt.plot(cube[:, 1, 3]); plt.show()  # timeseries
"""
import datetime as dt
import os.path
import re

import iris
import numpy as np
import pytest
from scmdata import ScmDataFrame

from netcdf_scm.iris_cube_wrappers import MarbleCMIP5Cube

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA = np.array(
    [
        [[30, 40, 50, 60], [110, 120, 190, 260], [3, 60, 20, 40]],
        [[0, 15, 45, 90], [300, 350, 450, 270], [10, 70, 90, 130]],
        [[60, 120, 60, 60], [510, 432, 220, 280], [50, 60, 55, 60]],
    ]
)

SURFACE_FRACS = np.array([[0, 30, 0, 10], [80, 100, 0, 50], [20, 10, 51, 15]])
AREA_WEIGHTS = np.array([[1.2, 1.2, 1.2, 1.2], [2, 2, 2, 2], [1.1, 1.1, 1.1, 1.1]])

TEST_RSDT_PATH = os.path.join(
    ROOT_DIR,
    "cmip5",
    "experiment",
    "Amon",
    "rsdt",
    "model",
    "realisation",
    "rsdt_Amon_model_experiment_realisation_185001-185003.nc",
)
TEST_GPP_PATH = os.path.join(
    ROOT_DIR,
    "cmip5",
    "experiment",
    "Lmon",
    "gpp",
    "model",
    "realisation",
    "gpp_Lmon_model_experiment_realisation_185001-185003.nc",
)
TEST_CSOILFAST_PATH = os.path.join(
    ROOT_DIR,
    "cmip5",
    "experiment",
    "Lmon",
    "cSoilFast",
    "model",
    "realisation",
    "cSoilFast_Lmon_model_experiment_realisation_185001-185003.nc",
)
TEST_HFDS_PATH = os.path.join(
    ROOT_DIR,
    "cmip5",
    "experiment",
    "Omon",
    "hfds",
    "model",
    "realisation",
    "hfds_Omon_model_experiment_realisation_185001-185003.nc",
)

TEST_AREACEALLA_PATH = os.path.join(
    ROOT_DIR,
    "cmip5",
    "experiment",
    "fx",  # TODO: Be careful when writing table id retrieval test as this is Ofx for CMIP6Output
    "areacella",
    "model",
    "r0i0p0",
    "areacella_fx_model_experiment_r0i0p0.nc",
)
TEST_AREACEALLO_PATH = os.path.join(
    ROOT_DIR,
    "cmip5",
    "experiment",
    "fx",
    "areacello",
    "model",
    "r0i0p0",
    "areacello_fx_model_experiment_r0i0p0.nc",
)

TEST_SFTLF_PATH = os.path.join(
    ROOT_DIR,
    "cmip5",
    "experiment",
    "fx",
    "sftlf",
    "model",
    "r0i0p0",
    "sftlf_fx_model_experiment_r0i0p0.nc",
)
TEST_SFTOF_PATH = os.path.join(
    ROOT_DIR,
    "cmip5",
    "experiment",
    "fx",
    "sftof",
    "model",
    "r0i0p0",
    "sftof_fx_model_experiment_r0i0p0.nc",
)

SCMDF_TIME = [
    dt.datetime(1850, 1, 16, 12),
    dt.datetime(1850, 2, 15, 0),
    dt.datetime(1850, 3, 16, 12),
]


LAND_EFFECTIVE_AREAS = (SURFACE_FRACS * AREA_WEIGHTS) / 100
OCEAN_EFFECTIVE_AREAS = ((100 - SURFACE_FRACS) * AREA_WEIGHTS) / 100
AREAS = {
    "World": np.sum(AREA_WEIGHTS),
    "World|Land": np.sum(LAND_EFFECTIVE_AREAS),
    "World|Ocean": np.sum(OCEAN_EFFECTIVE_AREAS),
    "World|Northern Hemisphere": np.sum(AREA_WEIGHTS[:2, :]),
    "World|Southern Hemisphere": np.sum(AREA_WEIGHTS[2, :]),
    "World|Northern Hemisphere|Land": np.sum(LAND_EFFECTIVE_AREAS[:2, :]),
    "World|Southern Hemisphere|Land": np.sum(LAND_EFFECTIVE_AREAS[2, :]),
    "World|Northern Hemisphere|Ocean": np.sum(OCEAN_EFFECTIVE_AREAS[:2, :]),
    "World|Southern Hemisphere|Ocean": np.sum(OCEAN_EFFECTIVE_AREAS[2, :]),
    "World|North Atlantic Ocean": 1,
    "World|El Nino N3.4": 2,
}


def _add_land_area_metadata(in_scmdf, realm):

    for region in in_scmdf["region"].unique():
        r_key = "area_{} (m**2)".format(
            region.lower().replace("|", "_").replace(" ", "_")
        )
        if region in (
            "World",
            "World|Northern Hemisphere",
            "World|Southern Hemisphere",
        ):
            if realm in ("land", "ocean"):
                in_scmdf.metadata[r_key] = AREAS[
                    "{}|{}".format(region, realm.capitalize())
                ]
                continue

        in_scmdf.metadata[r_key] = AREAS[region]

    return in_scmdf


def get_rsdt_expected_results():
    world_values = np.sum(np.sum(RAW_DATA * AREA_WEIGHTS, axis=2), axis=1) / np.sum(
        AREA_WEIGHTS
    )

    land_weights = SURFACE_FRACS * AREA_WEIGHTS
    world_land_values = np.sum(
        np.sum(RAW_DATA * land_weights, axis=2), axis=1
    ) / np.sum(land_weights)

    ocean_weights = (100 - SURFACE_FRACS) * AREA_WEIGHTS
    world_ocean_values = np.sum(
        np.sum(RAW_DATA * ocean_weights, axis=2), axis=1
    ) / np.sum(ocean_weights)

    nh_area_weights = np.copy(AREA_WEIGHTS)
    nh_area_weights[2, :] = 0
    world_nh_values = np.sum(
        np.sum(RAW_DATA * nh_area_weights, axis=2), axis=1
    ) / np.sum(nh_area_weights)

    sh_area_weights = np.copy(AREA_WEIGHTS)
    sh_area_weights[:2, :] = 0
    world_sh_values = np.sum(
        np.sum(RAW_DATA * sh_area_weights, axis=2), axis=1
    ) / np.sum(sh_area_weights)

    # we do these by hand: yes they're very slow but that's the point
    world_nh_land_values = np.array(
        [
            (40 * 30 + 60 * 10) * 1.2 + (110 * 80 + 120 * 100 + 260 * 50) * 2,
            (15 * 30 + 90 * 10) * 1.2 + (300 * 80 + 350 * 100 + 270 * 50) * 2,
            (120 * 30 + 60 * 10) * 1.2 + (510 * 80 + 432 * 100 + 280 * 50) * 2,
        ]
    ) / ((30 + 10) * 1.2 + (80 + 100 + 50) * 2)

    world_sh_land_values = np.array(
        [
            (3 * 20 + 60 * 10 + 20 * 51 + 40 * 15) * 1.1,
            (10 * 20 + 70 * 10 + 90 * 51 + 130 * 15) * 1.1,
            (50 * 20 + 60 * 10 + 55 * 51 + 60 * 15) * 1.1,
        ]
    ) / ((20 + 10 + 51 + 15) * 1.1)

    world_nh_ocean_values = np.array(
        [
            (30 * 100 + 40 * 70 + 50 * 100 + 60 * 90) * 1.2
            + (110 * 20 + 190 * 100 + 260 * 50) * 2,
            (0 * 100 + 15 * 70 + 45 * 100 + 90 * 90) * 1.2
            + (300 * 20 + 450 * 100 + 270 * 50) * 2,
            (60 * 100 + 120 * 70 + 60 * 100 + 60 * 90) * 1.2
            + (510 * 20 + 220 * 100 + 280 * 50) * 2,
        ]
    ) / ((100 + 70 + 100 + 90) * 1.2 + (20 + 100 + 50) * 2)

    world_sh_ocean_values = np.array(
        [
            (3 * 80 + 60 * 90 + 20 * 49 + 40 * 85) * 1.1,
            (10 * 80 + 70 * 90 + 90 * 49 + 130 * 85) * 1.1,
            (50 * 80 + 60 * 90 + 55 * 49 + 60 * 85) * 1.1,
        ]
    ) / ((80 + 90 + 49 + 85) * 1.1)

    world_north_atlantic_values = np.array([260, 270, 280])

    world_elnino_values = np.array([190, 450, 220])

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
            world_north_atlantic_values,
            world_elnino_values,
        ]
    ).T

    exp = ScmDataFrame(
        data=data,
        index=SCMDF_TIME,
        columns={
            "model": "unspecified",
            "scenario": "experiment",
            "region": [
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
            ],
            "variable": "rsdt",
            "unit": "W m^-2",
            "climate_model": "model",
            "activity_id": "cmip5",
            "member_id": "realisation",
            "variable_standard_name": "toa_incoming_shortwave_flux",
            "mip_era": "CMIP5",
        },
    )

    exp.metadata = {
        "calendar": "gregorian",
        "land_fraction": np.sum(AREA_WEIGHTS * SURFACE_FRACS)
        / (100 * np.sum(AREA_WEIGHTS)),
        "land_fraction_northern_hemisphere": np.sum(nh_area_weights * SURFACE_FRACS)
        / (100 * np.sum(nh_area_weights)),
        "land_fraction_southern_hemisphere": np.sum(sh_area_weights * SURFACE_FRACS)
        / (100 * np.sum(sh_area_weights)),
        "modeling_realm": "atmos",
        "Conventions": "CF-1.5",
        "crunch_source_files": "Files: ['/cmip5/experiment/Amon/rsdt/model/realisation/rsdt_Amon_model_experiment_realisation_185001-185003.nc']; areacella: ['/cmip5/experiment/fx/areacella/model/r0i0p0/areacella_fx_model_experiment_r0i0p0.nc']; sftlf: ['/cmip5/experiment/fx/sftlf/model/r0i0p0/sftlf_fx_model_experiment_r0i0p0.nc']",
    }
    exp = _add_land_area_metadata(exp, realm="atmos")

    return exp


def get_gpp_expected_results():
    land_weights = SURFACE_FRACS * AREA_WEIGHTS
    world_values = np.sum(np.sum(RAW_DATA * land_weights, axis=2), axis=1) / np.sum(
        land_weights
    )

    world_land_values = world_values

    nh_area_weights = np.copy(AREA_WEIGHTS)
    nh_area_weights[2, :] = 0
    # we do these by hand: yes they're very slow but that's the point
    world_nh_land_values = np.array(
        [
            (40 * 30 + 60 * 10) * 1.2 + (110 * 80 + 120 * 100 + 260 * 50) * 2,
            (15 * 30 + 90 * 10) * 1.2 + (300 * 80 + 350 * 100 + 270 * 50) * 2,
            (120 * 30 + 60 * 10) * 1.2 + (510 * 80 + 432 * 100 + 280 * 50) * 2,
        ]
    ) / ((30 + 10) * 1.2 + (80 + 100 + 50) * 2)
    world_nh_values = world_nh_land_values

    sh_area_weights = np.copy(AREA_WEIGHTS)
    sh_area_weights[:2, :] = 0
    world_sh_land_values = np.array(
        [
            (3 * 20 + 60 * 10 + 20 * 51 + 40 * 15) * 1.1,
            (10 * 20 + 70 * 10 + 90 * 51 + 130 * 15) * 1.1,
            (50 * 20 + 60 * 10 + 55 * 51 + 60 * 15) * 1.1,
        ]
    ) / ((20 + 10 + 51 + 15) * 1.1)
    world_sh_values = world_sh_land_values

    data = np.vstack(
        [
            world_values,
            world_land_values,
            world_nh_values,
            world_sh_values,
            world_nh_land_values,
            world_sh_land_values,
        ]
    ).T

    exp = ScmDataFrame(
        data=data,
        index=SCMDF_TIME,
        columns={
            "model": "unspecified",
            "scenario": "experiment",
            "region": [
                "World",
                "World|Land",
                "World|Northern Hemisphere",
                "World|Southern Hemisphere",
                "World|Northern Hemisphere|Land",
                "World|Southern Hemisphere|Land",
            ],
            "variable": "gpp",
            "unit": "kg m^-2 s^-1",
            "climate_model": "model",
            "activity_id": "cmip5",
            "member_id": "realisation",
            "variable_standard_name": "gross_primary_productivity_of_carbon",
            "mip_era": "CMIP5",
        },
    )
    exp.metadata = {
        "calendar": "gregorian",
        "modeling_realm": "land",
        "Conventions": "CF-1.5",
        "crunch_source_files": "Files: ['/cmip5/experiment/Lmon/gpp/model/realisation/gpp_Lmon_model_experiment_realisation_185001-185003.nc']; sftlf: ['/cmip5/experiment/fx/sftlf/model/r0i0p0/sftlf_fx_model_experiment_r0i0p0.nc']; areacella: ['/cmip5/experiment/fx/areacella/model/r0i0p0/areacella_fx_model_experiment_r0i0p0.nc']",
    }
    exp = _add_land_area_metadata(exp, realm="land")

    return exp


def get_csoilfast_expected_results():
    exp_scmdf = get_gpp_expected_results()
    exp_scmdf.set_meta("cSoilFast", "variable")
    exp_scmdf.set_meta("fast_soil_pool_carbon_content", "variable_standard_name")
    exp_scmdf.set_meta("kg m^-2", "unit")

    exp_scmdf.metadata[
        "crunch_source_files"
    ] = "Files: ['/cmip5/experiment/Lmon/cSoilFast/model/realisation/cSoilFast_Lmon_model_experiment_realisation_185001-185003.nc']; sftlf: ['/cmip5/experiment/fx/sftlf/model/r0i0p0/sftlf_fx_model_experiment_r0i0p0.nc']; areacella: ['/cmip5/experiment/fx/areacella/model/r0i0p0/areacella_fx_model_experiment_r0i0p0.nc']"
    exp_scmdf = _add_land_area_metadata(exp_scmdf, realm="land")

    return exp_scmdf


def get_hfds_expected_results():
    sftof_fracs = 100 - SURFACE_FRACS
    ocean_weights = sftof_fracs * AREA_WEIGHTS
    world_values = np.sum(np.sum(RAW_DATA * ocean_weights, axis=2), axis=1) / np.sum(
        ocean_weights
    )

    world_ocean_values = world_values

    nh_area_weights = np.copy(AREA_WEIGHTS)
    nh_area_weights[2, :] = 0
    # we do these by hand: yes they're very slow but that's the point
    world_nh_ocean_values = np.array(
        [
            (30 * 100 + 40 * 70 + 50 * 100 + 60 * 90) * 1.2
            + (110 * 20 + 190 * 100 + 260 * 50) * 2,
            (0 * 100 + 15 * 70 + 45 * 100 + 90 * 90) * 1.2
            + (300 * 20 + 450 * 100 + 270 * 50) * 2,
            (60 * 100 + 120 * 70 + 60 * 100 + 60 * 90) * 1.2
            + (510 * 20 + 220 * 100 + 280 * 50) * 2,
        ]
    ) / ((100 + 70 + 100 + 90) * 1.2 + (20 + 100 + 50) * 2)
    world_nh_values = world_nh_ocean_values

    sh_area_weights = np.copy(AREA_WEIGHTS)
    sh_area_weights[:2, :] = 0
    world_sh_ocean_values = np.array(
        [
            (3 * 80 + 60 * 90 + 20 * 49 + 40 * 85) * 1.1,
            (10 * 80 + 70 * 90 + 90 * 49 + 130 * 85) * 1.1,
            (50 * 80 + 60 * 90 + 55 * 49 + 60 * 85) * 1.1,
        ]
    ) / ((80 + 90 + 49 + 85) * 1.1)
    world_sh_values = world_sh_ocean_values

    world_north_atlantic_values = np.array([260, 270, 280])

    world_elnino_values = np.array([190, 450, 220])

    data = np.vstack(
        [
            world_values,
            world_ocean_values,
            world_nh_values,
            world_sh_values,
            world_nh_ocean_values,
            world_sh_ocean_values,
            world_north_atlantic_values,
            world_elnino_values,
        ]
    ).T

    exp = ScmDataFrame(
        data=data,
        index=SCMDF_TIME,
        columns={
            "model": "unspecified",
            "scenario": "experiment",
            "region": [
                "World",
                "World|Ocean",
                "World|Northern Hemisphere",
                "World|Southern Hemisphere",
                "World|Northern Hemisphere|Ocean",
                "World|Southern Hemisphere|Ocean",
                "World|North Atlantic Ocean",
                "World|El Nino N3.4",
            ],
            "variable": "hfds",
            "unit": "W m^-2",
            "climate_model": "model",
            "activity_id": "cmip5",
            "member_id": "realisation",
            "variable_standard_name": "surface_downward_heat_flux_in_sea_water",
            "mip_era": "CMIP5",
        },
    )
    exp.metadata = {
        "calendar": "gregorian",
        "modeling_realm": "ocean",
        "Conventions": "CF-1.5",
        "crunch_source_files": "Files: ['/cmip5/experiment/Omon/hfds/model/realisation/hfds_Omon_model_experiment_realisation_185001-185003.nc']; sftof: ['/cmip5/experiment/fx/sftof/model/r0i0p0/sftof_fx_model_experiment_r0i0p0.nc']; areacello: ['/cmip5/experiment/fx/areacello/model/r0i0p0/areacello_fx_model_experiment_r0i0p0.nc']",
    }
    exp = _add_land_area_metadata(exp, realm="ocean")

    return exp


@pytest.mark.parametrize(
    "test_data,invalid_regions,expected_results",
    [
        (TEST_RSDT_PATH, None, get_rsdt_expected_results()),
        (
            TEST_GPP_PATH,
            {
                "World|Ocean",
                "World|Northern Hemisphere|Ocean",
                "World|Southern Hemisphere|Ocean",
                "World|North Atlantic Ocean",
                "World|El Nino N3.4",
            },
            get_gpp_expected_results(),
        ),
        (
            TEST_CSOILFAST_PATH,
            {
                "World|Ocean",
                "World|Northern Hemisphere|Ocean",
                "World|Southern Hemisphere|Ocean",
                "World|North Atlantic Ocean",
                "World|El Nino N3.4",
            },
            get_csoilfast_expected_results(),
        ),
        (
            TEST_HFDS_PATH,
            {
                "World|Land",
                "World|Northern Hemisphere|Land",
                "World|Southern Hemisphere|Land",
            },
            get_hfds_expected_results(),
        ),
    ],
)
def test_scm_timeseries_crunching(
    assert_scmdata_frames_allclose, test_data, invalid_regions, expected_results
):
    tcube = MarbleCMIP5Cube()
    tcube.load_data_from_path(test_data)
    all_regions = {
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
    }
    regions = (
        list(all_regions - invalid_regions)
        if invalid_regions is not None
        else list(all_regions)
    )
    if invalid_regions is not None:
        for r in invalid_regions:
            error_msg = re.escape("All weights are zero for region: `{}`".format(r))
            with pytest.raises(ValueError, match=error_msg):
                tcube.get_scm_timeseries(regions=[r])

    res = tcube.get_scm_timeseries(regions=regions)
    assert_scmdata_frames_allclose(res, expected_results)


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
    write_surface_frac_file(
        TEST_SFTLF_PATH, lat, lon, "land_area_fraction", "sftlf", "%"
    )
    write_surface_frac_file(
        TEST_SFTOF_PATH, lat, lon, "sea_area_fraction", "sftof", "%", inverse=True
    )
    write_area_file(TEST_AREACEALLA_PATH, lat, lon, "cell_area", "areacella", "m^2")
    write_area_file(TEST_AREACEALLO_PATH, lat, lon, "cell_area", "areacello", "m^2")
    write_data_file(
        TEST_RSDT_PATH,
        lat,
        lon,
        "toa_incoming_shortwave_flux",
        "rsdt",
        "W m-2",
        {"modeling_realm": "atmos"},
    )
    write_data_file(
        TEST_GPP_PATH,
        lat,
        lon,
        "gross_primary_productivity_of_carbon",
        "gpp",
        "kg m-2 s-1",
        {"modeling_realm": "land"},
    )
    write_data_file(
        TEST_CSOILFAST_PATH,
        lat,
        lon,
        "fast_soil_pool_carbon_content",
        "cSoilFast",
        "kg m-2",
        {"modeling_realm": "land"},
    )
    write_data_file(
        TEST_HFDS_PATH,
        lat,
        lon,
        "surface_downward_heat_flux_in_sea_water",
        "hfds",
        "W m-2",
        {"modeling_realm": "ocean"},
    )


def write_surface_frac_file(
    write_path, lat, lon, standard_name, var_name, units, inverse=False
):
    data = SURFACE_FRACS if not inverse else 100 - SURFACE_FRACS
    cube = iris.cube.Cube(
        data,
        standard_name=standard_name,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(lat, 0), (lon, 1)],
    )
    save_cube_in_path(cube, write_path)


def write_area_file(write_path, lat, lon, standard_name, var_name, units):
    cube = iris.cube.Cube(
        AREA_WEIGHTS,
        standard_name=standard_name,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(lat, 0), (lon, 1)],
    )
    save_cube_in_path(cube, write_path)


def write_data_file(write_path, lat, lon, standard_name, var_name, units, attributes):
    time = iris.coords.DimCoord(
        np.array([15.5, 45, 74.5]),
        standard_name="time",
        units="days since 1850-01-01 00:00:00",
    )
    time.guess_bounds()

    cube = iris.cube.Cube(
        RAW_DATA,
        standard_name=standard_name,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )
    cube.attributes = attributes
    save_cube_in_path(cube, write_path)


def save_cube_in_path(cube, write_path):
    dir_to_save = os.path.dirname(write_path)
    if not os.path.isdir(dir_to_save):
        os.makedirs(dir_to_save)

    iris.save(cube, write_path)
