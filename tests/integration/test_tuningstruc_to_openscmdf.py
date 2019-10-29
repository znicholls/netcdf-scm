import datetime as dt
import re
from os.path import join

import numpy as np
import pandas as pd
import pytest
from conftest import TEST_DATA_ROOT_DIR
from scmdata import ScmDataFrame

from netcdf_scm.wranglers import (
    convert_scmdf_to_tuningstruc,
    convert_tuningstruc_to_scmdf,
    get_tuningstruc_name_from_df,
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
        {
            "location": join(TEST_DATA_TUNINGSTRUCS_DIR, "CABLE_rcp85_C0_npp.mat"),
            "var": "npp",
            "region": "World",
            "unit": "PgC/yr",
            "scenario": "RCP85",
            "climate_model_2100_values": {"CABLE": 30.967642},
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
    test_df = ScmDataFrame(
        np.array([1, 2, 3]),
        index=[dt.datetime(y, 1, 1) for y in [1990, 1991, 1992]],
        columns={
            "variable": "var",
            "region": "World",
            "unit": "K",
            "scenario": "test-scenario",
            "model": "test_model",
            "climate_model": "test_cm",
            "member_id": "tmember-id",
        },
    )

    convert_scmdf_to_tuningstruc(test_df, tmpdir, prefix="test_tuningstruc")
    expected_outfile = join(
        tmpdir, "test_tuningstruc_VAR_TEST-SCENARIO_TMEMBER-ID_WORLD.mat"
    )

    reread = convert_tuningstruc_to_scmdf(expected_outfile)
    assert (reread["unit"] == "K").all()


def test_convert_tuningstruc_to_scmdf_errors(test_file_info):
    test_file = test_file_info["location"]
    error_msg = r"Cannot determine \S* " + re.escape("from file: {}".format(test_file))
    with pytest.raises(KeyError, match=error_msg):
        convert_tuningstruc_to_scmdf(test_file)


def test_convert_scmdf_to_tuningstruc(test_file_info, tmpdir):
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
        join(
            tmpdir,
            "{}_{}_MEMBER-ID_{}".format(tvar, tscen, tregion)
            .replace(" ", "_")
            .replace("|", "_")
            .upper(),
        )
        + ".mat"
    )
    convert_scmdf_to_tuningstruc(start, tmpdir)
    res = convert_tuningstruc_to_scmdf(expected_outfile)

    pd.testing.assert_frame_equal(start.timeseries(), res.timeseries())


def test_get_tuningstruc_name_from_df_error():
    df = pd.DataFrame(["scen_1", "scen_2"], columns=["scenario"])
    with pytest.raises(ValueError, match="More than one scenario in df"):
        get_tuningstruc_name_from_df(df, "test", "filler")
