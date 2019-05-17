from os.path import join
import datetime as dt
import re


import pytest
import pandas as pd
import numpy as np
from openscm.scmdataframe import ScmDataFrame


from conftest import TEST_DATA_ROOT_DIR
from netcdf_scm.wranglers import (
    convert_tuningstruc_to_scmdf,
    convert_scmdf_to_tuningstruc,
)


TEST_DATA_TUNINGSTRUCS_DIR = join(TEST_DATA_ROOT_DIR, "tuningstrucs")


@pytest.fixture(
    scope="function",
    params=[
        {
            "location": join(
                TEST_DATA_TUNINGSTRUCS_DIR,
                "xavier_RCP85_GISSLE_transient_SMBdata_1850nan.mat",
            ),
            "var": "SMB",
            "region": "World",
            "unit": "mm",
            "scenario": "RCP85",
            "climate_model_2100_values": {
                "CCSM4": 47.64728531855956,
                "FGOALSG2": 92.01503047091413,
                "GISSE2R": 56.65268144044319,
                "IPSLCM5ALR": 124.97478670360111,
            },
        },
        {
            "location": join(TEST_DATA_TUNINGSTRUCS_DIR, "single_var_tuningstruc.mat"),
            "var": "cLitter",
            "region": "World",
            "unit": "GtC",
            "scenario": "RCP26",
            "climate_model_2100_values": {"BNUESM": 0.30397183},
        },
    ],
)
def test_file_info(request):
    yield request.param


@pytest.mark.parametrize("model", [None, "junk"])
def test_convert_tuningstruc_to_scmdf(test_file_info, model):
    test_file = test_file_info["location"]
    tkwargs = {"model": model}
    tvar = test_file_info["var"]
    tregion = test_file_info["region"]
    tunit = test_file_info["unit"]
    tscen = test_file_info["scenario"]

    res = convert_tuningstruc_to_scmdf(
        test_file, tvar, tregion, tunit, tscen, **tkwargs
    )

    if model is None:
        assert (res["model"] == "unspecified").all()  # default
    else:
        assert (res["model"] == model).all()

    for cm, v in test_file_info["climate_model_2100_values"].items():
        rv = (
            res.filter(
                climate_model=cm,
                year=2100,
                variable=tvar,
                region=tregion,
                scenario=tscen,
                unit=tunit,
            )
            .timeseries()
            .values
        )
        np.testing.assert_allclose(rv, v)


def test_convert_scmdf_to_tuningstruc_single_char_unit(tmpdir):
    tbase = join(tmpdir, "test_tuningstruc")

    test_df = ScmDataFrame(
        np.array([1, 2, 3]),
        index=[dt.datetime(y, 1, 1) for y in [1990, 1991, 1992]],
        columns={
            "variable": "var",
            "region": "World",
            "unit": "K",
            "scenario": "test",
            "model": "test",
            "climate_model": "test",
        },
    )

    convert_scmdf_to_tuningstruc(test_df, tbase)
    expected_outfile = join(tmpdir, "test_tuningstruc_test_test_var_World.mat")

    reread = convert_tuningstruc_to_scmdf(expected_outfile)
    assert (reread["unit"] == "K").all()


def test_convert_tuningstruc_to_scmdf_errors(test_file_info):
    test_file = test_file_info["location"]
    error_msg = r"Cannot determine \S* " + re.escape("from file: {}".format(test_file))
    with pytest.raises(KeyError, match=error_msg):
        convert_tuningstruc_to_scmdf(test_file)


def test_convert_scmdf_to_tuningstruc(test_file_info, tmpdir):
    tbase = join(tmpdir, "test_tuningstruc")

    test_file = test_file_info["location"]
    tvar = test_file_info["var"]
    tregion = "World|Northern Hemisphere|Ocean"
    tunit = test_file_info["unit"]
    tscen = test_file_info["scenario"]
    tmodel = "iam"
    start = convert_tuningstruc_to_scmdf(
        test_file, tvar, tregion, tunit, tscen, model=tmodel
    )

    expected_outfile = (
        "{}_{}_{}_{}_{}.mat".format(tbase, tscen, tmodel, tvar, tregion)
        .replace(" ", "_")
        .replace("|", "_")
    )
    convert_scmdf_to_tuningstruc(start, tbase)
    res = convert_tuningstruc_to_scmdf(expected_outfile)

    pd.testing.assert_frame_equal(start.timeseries(), res.timeseries())
