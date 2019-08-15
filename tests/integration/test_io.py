import os

import numpy as np

import netcdf_scm
from netcdf_scm.io import load_scmdataframe, save_netcdf_scm_nc
from netcdf_scm.iris_cube_wrappers import CMIP6OutputCube


def _assert_scm_dataframe(scmdf, expected, **kwargs):
    d = scmdf.filter(**kwargs).timeseries()
    assert not d.empty
    np.testing.assert_allclose(d.values.squeeze(), expected)


def test_load_scmdataframe(test_data_netcdfscm_nc_file):
    loaded = load_scmdataframe(test_data_netcdfscm_nc_file)
    assert (loaded["scenario"] == "rcp45").all()
    assert (loaded["climate_model"] == "ACCESS1-0").all()
    assert (loaded["variable"] == "tas").all()
    assert (loaded["variable_standard_name"] == "air_temperature").all()
    assert (loaded["unit"] == "K").all()
    assert (loaded["member_id"] == "r1i1p1").all()
    assert (loaded["mip_era"] == "CMIP5").all()
    assert (loaded["activity_id"] == "cmip5").all()

    _assert_scm_dataframe(loaded, 285.521667, region="World", year=2006, month=1)
    _assert_scm_dataframe(loaded, 279.19043, region="World|Land", year=2019, month=3)
    _assert_scm_dataframe(
        loaded, 287.103729, region="World|Northern Hemisphere", year=2032, month=11
    )
    _assert_scm_dataframe(
        loaded,
        290.850189,
        region="World|Northern Hemisphere|Ocean",
        year=2049,
        month=12,
    )

    assert (
        loaded.metadata["crunch_netcdf_scm_version"]
        == "1.0.0+97.g6d5c5ae (more info at github.com/znicholls/netcdf-scm)"
    )
    assert (
        loaded.metadata["institution"]
        == "CSIRO (Commonwealth Scientific and Industrial Research Organisation, Australia), and BOM (Bureau of Meteorology, Australia)"
    )
    assert (
        loaded.metadata["title"] == "ACCESS1-0 model output prepared for CMIP5 RCP4.5"
    )
    np.testing.assert_allclose(
        loaded.metadata["land_fraction_northern_hemisphere"], 0.38912639
    )


def test_save_cube_and_load_scmdataframe(tmpdir, test_cmip6_output_file):
    base = CMIP6OutputCube()
    base.load_data_from_path(test_cmip6_output_file)
    out_file = os.path.join(tmpdir, "test_save_file.nc")

    save_netcdf_scm_nc(base.get_scm_timeseries_cubes(), out_file)

    loaded = load_scmdataframe(out_file)
    assert (loaded["scenario"] == "1pctCO2").all()
    assert (loaded["climate_model"] == "BCC-CSM2-MR").all()
    assert (loaded["variable"] == "rlut").all()
    assert (loaded["variable_standard_name"] == "toa_outgoing_longwave_flux").all()
    assert (loaded["unit"] == "W m^-2").all()
    assert (loaded["activity_id"] == "CMIP").all()
    assert (loaded["member_id"] == "r1i1p1f1").all()
    assert (loaded["mip_era"] == "CMIP6").all()
    assert (loaded["activity_id"] == "CMIP").all()

    _assert_scm_dataframe(loaded, 236.569464, region="World", year=1859, month=12)
    _assert_scm_dataframe(loaded, 243.072575, region="World|Ocean", year=1856, month=10)
    _assert_scm_dataframe(
        loaded, 235.025871, region="World|Southern Hemisphere", year=1853, month=6
    )
    _assert_scm_dataframe(
        loaded, 234.333421, region="World|Southern Hemisphere|Land", year=1850, month=1
    )

    assert loaded.metadata[
        "crunch_netcdf_scm_version"
    ] == "{} (more info at github.com/znicholls/netcdf-scm)".format(
        netcdf_scm.__version__
    )
    assert (
        loaded.metadata["institution"]
        == "Beijing Climate Center, Beijing 100081, China"
    )
    assert loaded.metadata["title"] == "BCC-CSM2-MR output prepared for CMIP6"
    np.testing.assert_allclose(
        loaded.metadata["land_fraction_northern_hemisphere"], 0.38681185060261924
    )
    assert (
        loaded.metadata["source"]
        == "BCC-CSM 2 MR (2017):   aerosol: none  atmos: BCC_AGCM3_MR (T106; 320 x 160 longitude/latitude; 46 levels; top level 1.46 hPa)  atmosChem: none  land: BCC_AVIM2  landIce: none  ocean: MOM4 (1/3 deg 10S-10N, 1/3-1 deg 10-30 N/S, and 1 deg in high latitudes; 360 x 232 longitude/latitude; 40 levels; top grid cell 0-10 m)  ocnBgchem: none  seaIce: SIS2"
    )
