import datetime as dt

from netcdf_scm.misc_readers import read_cmip6_concs_gmnhsh


def test_conc_gmnhsh_reading(test_data_conc_gmnhsh_file):
    res = read_cmip6_concs_gmnhsh(test_data_conc_gmnhsh_file)

    expected_columns = {
        "climate_model": "MAGICC7",
        "mip_era": "CMIP6",
        "model": "unspecified",
    }

    if "ssp370" in test_data_conc_gmnhsh_file:
        assert res["time"].min() == dt.datetime(2015, 1, 16, 12, 0)
        assert res["time"].max() == dt.datetime(2500, 12, 16, 12, 0)
        expected_columns = {
            "variable": "mole_fraction_of_c4f10_in_air",
            "scenario": "ssp370-lowNTCF",
            "model": "AIM",
            "activity_id": "input4MIPs",
            "unit": "ppt",
        }
    elif "ssp245" in test_data_conc_gmnhsh_file:
        assert res["time"].min() == dt.datetime(2015, 1, 16, 12, 0)
        assert res["time"].max() == dt.datetime(2500, 12, 16, 12, 0)
        expected_columns = {
            "variable": "mole_fraction_of_carbon_dioxide_in_air",
            "scenario": "ssp245",
            "model": "MESSAGE-GLOBIOM",
            "activity_id": "input4MIPs",
            "unit": "ppm",
        }
    else:
        assert res["time"].min() == dt.datetime(1, 1, 17, 12, 0)
        assert res["time"].max() == dt.datetime(2014, 12, 17, 12, 0)
        expected_columns = {
            "variable": "mole_fraction_of_carbon_dioxide_in_air",
            "scenario": "historical",
            "model": "unspecified",
            "activity_id": "input4MIPs",
            "unit": "ppm",
        }

    for k, v in expected_columns.items():
        assert (res[k] == v).all()

    assert res["variable_standard_name"].isnull().all()
    assert sorted(res["region"]) == [
        "World",
        "World|Northern Hemisphere",
        "World|Southern Hemisphere",
    ]
