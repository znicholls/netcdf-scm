import os

import numpy as np

from netcdf_scm.io import load_scmdataframe, save_netcdf_scm_nc
from netcdf_scm.iris_cube_wrappers import CMIP6OutputCube

from conftest import (
    TEST_DATA_NETCDFSCM_NC_FILE,
    TEST_CMIP6_OUTPUT_FILE,
    tdata_required
)


def _assert_scm_dataframe(scmdf, expected, **kwargs):
    d = scmdf.filter(**kwargs).timeseries()
    assert not d.empty
    np.testing.assert_allclose(d.values.squeeze(), expected)


@tdata_required
def test_load_scmdataframe():
    loaded = load_scmdataframe(TEST_DATA_NETCDFSCM_NC_FILE)
    assert (loaded["scenario"] == "rcp85").all()
    assert (loaded["climate_model"] == "NorESM1-ME").all()
    assert (loaded["variable"] == "surface_temperature").all()
    assert (loaded["unit"] == "K").all()
    assert (loaded["member_id"] == "r1i1p1").all()
    assert (loaded["mip"] == "CMIP5").all()

    _assert_scm_dataframe(
        loaded,
        285.06777954,
        region="World",
        year=2006,
        month=1
    )
    _assert_scm_dataframe(
        loaded,
        281.885468,
        region="World|Land",
        year=2006,
        month=1
    )
    _assert_scm_dataframe(
        loaded,
        296.85611,
        region="World|Northern Hemisphere",
        year=2006,
        month=1
    )
    _assert_scm_dataframe(
        loaded,
        293.116852,
        region="World|Northern Hemisphere|Ocean",
        year=2006,
        month=1
    )

    assert loaded.metadata["crunch_netcdf_scm_version"] == "1.0.0+77.g426a601 (more info at github.com/znicholls/netcdf-scm)"
    assert loaded.metadata["institution"] == "Norwegian Climate Centre"
    assert loaded.metadata["title"] == "NorESM1-ME model output prepared for CMIP5 RCP8.5"
    assert loaded.metadata["land_fraction_northern_hemisphere"] == 0.3839148322659226


@tdata_required
def test_save_cube_and_load_scmdataframe(tmpdir):
    base = CMIP6OutputCube()
    base.load_data_from_path(TEST_CMIP6_OUTPUT_FILE)
    out_file = os.path.join(tmpdir, "test_save_file.nc")

    save_netcdf_scm_nc(base.get_scm_timeseries_cubes(), out_file)

    loaded = load_scmdataframe(out_file)
    assert (loaded["scenario"] == "1pctCO2").all()
    assert (loaded["climate_model"] == "BCC-CSM2-MR").all()
    assert (loaded["variable"] == "toa_outgoing_longwave_flux").all()
    assert (loaded["unit"] == "W m^-2").all()
    assert (loaded["activity_id"] == "CMIP").all()
    assert (loaded["member_id"] == "r1i1p1f1").all()
    assert (loaded["mip"] == "CMIP6").all()

    _assert_scm_dataframe(
        loaded,
        285.06777954,
        region="World",
        year=2006,
        month=1
    )
    _assert_scm_dataframe(
        loaded,
        281.885468,
        region="World|Ocean",
        year=2006,
        month=1
    )
    _assert_scm_dataframe(
        loaded,
        296.85611,
        region="World|Southern Hemisphere",
        year=2006,
        month=1
    )
    _assert_scm_dataframe(
        loaded,
        293.116852,
        region="World|Southern Hemisphere|Land",
        year=2006,
        month=1
    )

    assert loaded.metadata["crunch_netcdf_scm_version"] == "{} (more info at github.com/znicholls/netcdf-scm)".format(netcdf_scm.__version__)
    assert loaded.metadata["institution"] == "Beijing Climate Center, Beijing 100081, China"
    assert loaded.metadata["title"] == "BCC-CSM2-MR output prepared for CMIP6"
    assert loaded.metadata["land_fraction_northern_hemisphere"] == 0.38029161
    assert loaded.metadata["source"] == "BCC-CSM 2 MR (2017):   aerosol: none  atmos: BCC_AGCM3_MR (T106; 320 x 160 longitude/latitude; 46 levels; top level 1.46 hPa)  atmosChem: none  land: BCC_AVIM2  landIce: none  ocean: MOM4 (1/3 deg 10S-10N, 1/3-1 deg 10-30 N/S, and 1 deg in high latitudes; 360 x 232 longitude/latitude; 40 levels; top grid cell 0-10 m)  ocnBgchem: none  seaIce: SIS2"
