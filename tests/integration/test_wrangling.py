from glob import glob
from os.path import isdir, join

from click.testing import CliRunner
from conftest import TEST_DATA_CMIP6_CRUNCH_OUTPUT

import netcdf_scm
from netcdf_scm.cli import wrangle_netcdf_scm_ncs


def test_wrangling_defaults(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_CMIP6_CRUNCH_OUTPUT
    OUTPUT_DIR = str(join(tmpdir, "new-sub-dir"))

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(wrangle_netcdf_scm_ncs, [INPUT_DIR, OUTPUT_DIR, "test-defaults"])
    assert result.exit_code == 0

    assert "netcdf-scm: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory: {}".format(OUTPUT_DIR) in result.output

    assert "rsut" in result.output
    assert "rlut" in result.output

    assert isdir(
        join(OUTPUT_DIR, "cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/1pctCO2/r1i1p1f1/Amon/rlut")
    )
    assert isdir(
        join(OUTPUT_DIR, "cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/1pctCO2/r1i1p1f1/Amon/rsut")
    )
    assert isdir(join(OUTPUT_DIR, "cmip6/CMIP6/ScenarioMIP/IPSL"))
    assert isdir(join(OUTPUT_DIR, "cmip6/CMIP6/ScenarioMIP/MRI"))


def test_wrangling_flat_blend_models(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_CMIP6_CRUNCH_OUTPUT
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                INPUT_DIR,
                OUTPUT_DIR,
                "test",
                "--flat",
                "--drs",
                "CMIP6Output",
                "--out-format",
                "tuningstrucs-blend-model",
            ],
        )
    assert result.exit_code == 0, result.output

    assert "netcdf-scm: {}".format(netcdf_scm.__version__) in result.output

    assert "Wrangling re.compile" in result.output
    assert ".*" in result.output
    assert ".mat" in result.output

    assert len(glob(join(OUTPUT_DIR, "*.mat"))) == 27


def test_wrangling_handles_integer_units(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_CMIP6_CRUNCH_OUTPUT
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                INPUT_DIR,
                OUTPUT_DIR,
                "test",
                "--regexp",
                ".*lai.*",
                "--out-format",
                "tuningstrucs-blend-model",
                "--flat",
                "--drs",
                "CMIP6Output",
            ],
        )
    assert result.exit_code == 0

    assert "netcdf-scm: {}".format(netcdf_scm.__version__) in result.output

    assert "lai" in result.output


def test_wrangling_force(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_CMIP6_CRUNCH_OUTPUT
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [INPUT_DIR, OUTPUT_DIR, "test", "--regexp", ".*lai.*", "-f", "--prefix", "test-prefix"],
    )
    assert result.exit_code == 0

    caplog.clear()
    with caplog.at_level("INFO"):
        result_skip = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [INPUT_DIR, OUTPUT_DIR, "--regexp", ".*lai.*", "--prefix", "test-prefix"],
        )
    assert result_skip.exit_code == 0

    skip_str_file = "Skipped (already exists, not overwriting) {}".format(
        join(
            OUTPUT_DIR,
            "cmip6/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r2i1p1f2/Lmon/lai/gr/v20181126/netcdf-scm_lai_Lmon_CNRM-CM6-1_historical_r2i1p1f2_gr_185001-201412.MAG",
        )
    )
    assert skip_str_file in result_skip.output

    caplog.clear()
    with caplog.at_level("INFO"):
        result_force = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                INPUT_DIR,
                OUTPUT_DIR,
                "test",
                "--regexp",
                ".*lai.*",
                "-f",
                "--prefix",
                "test-prefix",
            ],
        )
    assert result_force.exit_code == 0
    assert skip_str_file not in result_force.output


def test_wrangling_force_flat(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_CMIP6_CRUNCH_OUTPUT
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "test",
            "--regexp",
            ".*lai.*",
            "-f",
            "--flat",
            "--drs",
            "CMIP6Output",
            "--out-format",
            "tuningstrucs-blend-model",
        ],
    )
    assert result.exit_code == 0

    caplog.clear()
    with caplog.at_level("INFO"):
        result_skip = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                INPUT_DIR,
                OUTPUT_DIR,
                "test",
                "--regexp",
                ".*lai.*",
                "--flat",
                "--drs",
                "CMIP6Output",
                "--out-format",
                "tuningstrucs-blend-model",
            ],
        )
    assert result_skip.exit_code == 0

    skip_str_file = "Skipped (already exists, not overwriting) {}".format(
        join(
            OUTPUT_DIR, "LAI_HISTORICAL_R1I1P1F2_WORLD.mat"
        )
    )
    assert skip_str_file in result_skip.output

    result_force = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "test",
            "--regexp",
            ".*lai.*",
            "-f",
            "--flat",
            "--drs",
            "CMIP6Output",
            "--out-format",
            "tuningstrucs-blend-model",
        ],
    )
    assert result_force.exit_code == 0
    assert skip_str_file not in result_force.output


def test_wrangling_blended_models_default_drs_error(tmpdir):
    INPUT_DIR = TEST_DATA_CMIP6_CRUNCH_OUTPUT
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [INPUT_DIR, OUTPUT_DIR, "test", "--flat", "--out-format", "tuningstrucs-blend-model"],
    )
    assert result.exit_code != 0


def test_wrangling_blended_models_not_flat_error(tmpdir):
    INPUT_DIR = TEST_DATA_CMIP6_CRUNCH_OUTPUT
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [INPUT_DIR, OUTPUT_DIR, "test", "--nested", "--out-format", "tuningstrucs-blend-model"],
    )

    assert result.exit_code != 0
