import datetime as dt


from netcdf_scm.utils import read_cmip6_concs_gmnhsh


def test_conc_gmnhsh_reading(test_data_conc_gmnhsh_file):
    res =read_cmip6_concs_gmnhsh(test_data_conc_gmnhsh_file)

    assert res["variable"] == "Atmospheric Concentrations|CO2"
    assert sorted(res["region"]) == ["World", "World|Northern Hemisphere", "World|Southern Hemisphere"]
    assert res.time.max() == dt.datetime(2014, 12, 7)
    assert res.time.min() == dt.datetime(5, 12, 7)
