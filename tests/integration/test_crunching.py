from os import walk
from os.path import join, isfile, isdir


from click.testing import CliRunner
import pandas as pd
import numpy as np
from openscm.scmdataframe import ScmDataFrame

import netcdf_scm
from netcdf_scm.cli import crunch_data


from conftest import TEST_DATA_KNMI_DIR, TEST_DATA_MARBLE_CMIP5_DIR


def test_crunching(tmpdir):
    INPUT_DIR = TEST_DATA_MARBLE_CMIP5_DIR
    OUTPUT_DIR = str(tmpdir)
    VAR_TO_CRUNCH = ".*tas.*"

    runner = CliRunner()
    result = runner.invoke(
        crunch_data,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--cube-type",
            "MarbleCMIP5",
            "--var-to-crunch",
            VAR_TO_CRUNCH,
            "-f",
        ],
    )
    assert result.exit_code == 0
    assert "NetCDF SCM version: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory:" in result.output

    THRESHOLD_PERCENTAGE_DIFF = 10 ** -1
    files_found = 0
    for dirpath, dirnames, filenames in walk(OUTPUT_DIR):
        if not dirnames:
            assert len(filenames) == 1
            filename = filenames[0]
            files_found += 1

            knmi_data_name = "global_{}.dat".format("_".join(filename.split("_")[1:6]))
            knmi_data_path = join(TEST_DATA_KNMI_DIR, knmi_data_name)

            if not isfile(knmi_data_path):
                print("No data available for {}".format(knmi_data_path))
                continue

            knmi_data = pd.read_csv(
                knmi_data_path,
                skiprows=3,
                delim_whitespace=True,
                header=None,
                names=["year", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            ).melt(id_vars="year", var_name="month")
            knmi_data["year"] = knmi_data["year"].astype(int)
            knmi_data["month"] = knmi_data["month"].astype(int)
            knmi_data = knmi_data.set_index(["year", "month"])

            crunched_data = ScmDataFrame(join(dirpath, filename))
            comparison_data = (
                crunched_data.filter(region="World")
                .timeseries()
                .stack()
                .to_frame()
                .reset_index()[["time", 0]]
            )
            comparison_data = comparison_data.rename({0: "value"}, axis="columns")

            comparison_data["year"] = comparison_data["time"].apply(lambda x: x.year)
            comparison_data["month"] = comparison_data["time"].apply(lambda x: x.month)

            comparison_data = comparison_data.drop("time", axis="columns")
            comparison_data = comparison_data.set_index(["year", "month"])

            rel_difference = (knmi_data - comparison_data) / knmi_data
            # drop regions where times are not equal
            rel_difference = rel_difference.dropna()
            assert not rel_difference.empty, "not testing anything"

            assert_message = "{} data is not the same to within {}%".format(
                filename, THRESHOLD_PERCENTAGE_DIFF
            )
            all_close = (
                np.abs(rel_difference.values) < THRESHOLD_PERCENTAGE_DIFF / 100
            ).all()
            assert all_close, assert_message

            print(
                "{} file matches KNMI data to within {}%".format(
                    filename, THRESHOLD_PERCENTAGE_DIFF
                )
            )

    assert files_found == 5


def test_crunching_arguments(tmpdir):
    INPUT_DIR = TEST_DATA_MARBLE_CMIP5_DIR
    OUTPUT_DIR = str(tmpdir)
    VAR_TO_CRUNCH = ".*fco2antt.*"
    DATA_SUB_DIR = "custom-name"
    LAND_MASK_TRESHHOLD = 45

    runner = CliRunner()
    result = runner.invoke(
        crunch_data,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--cube-type",
            "MarbleCMIP5",
            "--var-to-crunch",
            VAR_TO_CRUNCH,
            "--data-sub-dir",
            DATA_SUB_DIR,
            "--land-mask-threshold",
            LAND_MASK_TRESHHOLD,
            "-f",
        ],
    )
    assert result.exit_code == 0

    assert "NetCDF SCM version: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory:" in result.output

    assert "fco2antt" in result.output
    assert "tas" not in result.output

    assert "Making output directory:" in result.output
    assert isdir(join(OUTPUT_DIR, DATA_SUB_DIR, "cmip5"))

    assert "land-mask-threshold: {}".format(LAND_MASK_TRESHHOLD) in result.output

    result_skip = runner.invoke(
        crunch_data,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--cube-type",
            "MarbleCMIP5",
            "--var-to-crunch",
            VAR_TO_CRUNCH,
            "--data-sub-dir",
            DATA_SUB_DIR,
            "--land-mask-threshold",
            LAND_MASK_TRESHHOLD,
        ],
    )
    assert result_skip.exit_code == 0

    skip_str = (
        "Skipped (already exist, not overwriting)\n"
        "========================================\n"
        "- {}".format(
            join(
                OUTPUT_DIR,
                DATA_SUB_DIR,
                "cmip5",
                "1pctCO2",
                "Amon",
                "fco2antt",
                "CanESM2",
                "r1i1p1",
                "netcdf-scm_fco2antt_Amon_CanESM2_1pctCO2_r1i1p1_185001-198912.csv",
            )
        )
    )
    assert skip_str in result_skip.output


def test_crunching_other_cube(tmpdir):
    INPUT_DIR = TEST_DATA_MARBLE_CMIP5_DIR
    OUTPUT_DIR = str(tmpdir)
    CUBE = "CMIP6Output"

    runner = CliRunner()
    result = runner.invoke(crunch_data, [INPUT_DIR, OUTPUT_DIR, "--cube-type", CUBE])
    assert result.exit_code  # non-zero exit code

    assert "cube-type: {}".format(CUBE) in result.output
