from glob import glob
from os import walk
from os.path import isdir, isfile, join

import numpy as np
import pandas as pd
from click.testing import CliRunner
from openscm.scmdataframe import ScmDataFrame

import netcdf_scm
from conftest import TEST_DATA_KNMI_DIR, TEST_DATA_MARBLE_CMIP5_DIR
from netcdf_scm.cli import crunch_data


def test_crunching(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_MARBLE_CMIP5_DIR
    OUTPUT_DIR = str(tmpdir)
    VAR_TO_CRUNCH = ".*tas.*"

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        crunch_data,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--cube-type",
            "MarbleCMIP5",
            "--regexp",
            VAR_TO_CRUNCH,
            "-f",
        ],
    )
    assert result.exit_code == 0
    assert "netcdf-scm: {}".format(netcdf_scm.__version__) in caplog.messages
    assert "Making output directory: {}/netcdf-scm-crunched".format(OUTPUT_DIR) in caplog.messages

    # Check that there is a log file  which contains 'INFO' log messages
    log_fnames = glob(join(OUTPUT_DIR, 'netcdf-scm-crunched', '*.log'))
    assert len(log_fnames) == 1

    with open(log_fnames[0]) as fh:
        log_file = fh.read()
        assert 'DEBUG' in log_file
    # Check that the logs are also written to stderr
    assert 'DEBUG' not in result.stderr
    assert 'INFO' in result.stderr

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


def test_crunching_arguments(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_MARBLE_CMIP5_DIR
    OUTPUT_DIR = str(tmpdir)
    VAR_TO_CRUNCH = ".*fco2antt.*"
    DATA_SUB_DIR = "custom-name"
    LAND_MASK_TRESHHOLD = 45

    runner = CliRunner()
    with caplog.at_level('INFO'):
        result = runner.invoke(
            crunch_data,
            [
                INPUT_DIR,
                OUTPUT_DIR,
                "--cube-type",
                "MarbleCMIP5",
                "--regexp",
                VAR_TO_CRUNCH,
                "--data-sub-dir",
                DATA_SUB_DIR,
                "--land-mask-threshold",
                LAND_MASK_TRESHHOLD,
                "-f",
            ],
        )
    assert result.exit_code == 0

    assert "netcdf-scm: {}".format(netcdf_scm.__version__) in caplog.messages
    assert "Making output directory: {}/custom-name".format(OUTPUT_DIR) in caplog.messages

    assert "fco2antt" in caplog.text
    assert "tas" not in caplog.text

    assert isdir(join(OUTPUT_DIR, DATA_SUB_DIR, "cmip5"))

    assert "land_mask_threshold: {}".format(LAND_MASK_TRESHHOLD) in caplog.text

    caplog.clear()

    with caplog.at_level('INFO'):
        result_skip = runner.invoke(
            crunch_data,
            [
                INPUT_DIR,
                OUTPUT_DIR,
                "--cube-type",
                "MarbleCMIP5",
                "--regexp",
                VAR_TO_CRUNCH,
                "--data-sub-dir",
                DATA_SUB_DIR,
                "--land-mask-threshold",
                LAND_MASK_TRESHHOLD,
            ],
        )
    assert result_skip.exit_code == 0

    skip_str = (
        "Skipped (already exist, not overwriting) {}".format(
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
    assert skip_str in caplog.text


def test_crunching_other_cube(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_MARBLE_CMIP5_DIR
    OUTPUT_DIR = str(tmpdir)
    CUBE = "CMIP6Output"

    runner = CliRunner()
    with caplog.at_level('INFO'):
        result = runner.invoke(crunch_data, [INPUT_DIR, OUTPUT_DIR, "--cube-type", CUBE])
    assert result.exit_code  # non-zero exit code

    assert "cube-type: {}".format(CUBE) in caplog.text
