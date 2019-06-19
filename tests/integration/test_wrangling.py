from glob import glob
from os.path import isdir, join

from click.testing import CliRunner
from conftest import TEST_DATA_CMIP6_CRUNCH_OUTPUT, TEST_DATA_MARBLE_CMIP5_CRUNCH_OUTPUT

import netcdf_scm
from netcdf_scm.cli import wrangle_netcdf_scm_ncs


def test_wrangling_defaults(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_CMIP6_CRUNCH_OUTPUT
    OUTPUT_DIR = str(join(tmpdir, "new-sub-dir"))
    test_wrangler = "test-defaults"

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [INPUT_DIR, OUTPUT_DIR, test_wrangler, "--drs", "CMIP6Output"],
        )
    assert result.exit_code == 0

    assert "netcdf-scm: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory: {}".format(OUTPUT_DIR) in result.output

    assert "rlut" in result.output
    assert "lai" in result.output
    assert "cSoilFast" in result.output

    assert isdir(
        join(
            OUTPUT_DIR,
            "CMIP6/CMIP/BCC/BCC-CSM2-MR/1pctCO2/r1i1p1f1/Amon/rlut/gn/v20181015",
        )
    )
    assert isdir(
        join(
            OUTPUT_DIR,
            "CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp126/r1i1p1f1/Lmon/cSoilFast/gr/v20190121",
        )
    )
    assert isdir(join(OUTPUT_DIR, "CMIP6/CMIP/CNRM-CERFACS"))

    with open(
        join(
            OUTPUT_DIR,
            "CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r1i1p1f2/Lmon/lai/gr/v20180917/netcdf-scm_lai_Lmon_CNRM-CM6-1_historical_r1i1p1f2_gr_200001-201412.MAG",
        )
    ) as f:
        content = f.read()

    assert "Contact: {}".format(test_wrangler) in content

    with open(
        join(
            OUTPUT_DIR,
            "flat/netcdf-scm_lai_Lmon_CNRM-CM6-1_historical_r1i1p1f2_gr_200001-201412.MAG",
        )
    ) as f:
        content = f.read()

    assert "Making symlink to {}".format(join(OUTPUT_DIR, "flat")) in result.output
    assert "Contact: {}".format(test_wrangler) in content


def test_wrangling_magicc_input_files(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_MARBLE_CMIP5_CRUNCH_OUTPUT
    OUTPUT_DIR = str(join(tmpdir, "new-sub-dir"))
    test_wrangler = "test wrangling magicc input point end of year files <email>"
    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                INPUT_DIR,
                OUTPUT_DIR,
                test_wrangler,
                "--out-format",
                "magicc-input-files-point-end-of-year",
                "--regexp",
                ".*tas.*",
                "--drs",
                "MarbleCMIP5",
            ],
        )
    assert result.exit_code == 0

    assert "netcdf-scm: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory: {}".format(OUTPUT_DIR) in result.output

    assert isdir(join(OUTPUT_DIR, "cmip5/1pctCO2/Amon/tas/CanESM2/r1i1p1"))
    assert isdir(join(OUTPUT_DIR, "cmip5/rcp26/Amon/tas/bcc-csm1-1/r1i1p1"))
    assert isdir(join(OUTPUT_DIR, "cmip5/rcp45/Amon/tas/HadCM3/r1i1p1"))
    assert isdir(join(OUTPUT_DIR, "cmip5/rcp45/Amon/tas/ACCESS1-0/r1i1p1"))
    assert isdir(join(OUTPUT_DIR, "cmip5/rcp85/Amon/tas/NorESM1-ME/r1i1p1"))

    with open(
        join(
            OUTPUT_DIR,
            "cmip5/rcp45/Amon/tas/HadCM3/r1i1p1/TAS_RCP45_HADCM3_R1I1P1_2006-2036_GLOBAL_SURFACE_TEMP.IN",
        )
    ) as f:
        content = f.read()

    assert "Contact: {}".format(test_wrangler) in content
    assert "timeseriestype: POINT_END_OF_YEAR" in content
    assert "2006" in content
    assert "2006." not in content
    assert "2030" in content
    assert "2030." not in content
    assert "2035" in content
    assert "2035." not in content
    assert "2036" not in content
    assert "2036." not in content


def test_wrangling_magicc_input_files_error(tmpdir, caplog):
    INPUT_DIR = TEST_DATA_CMIP6_CRUNCH_OUTPUT
    OUTPUT_DIR = str(join(tmpdir, "new-sub-dir"))

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                INPUT_DIR,
                OUTPUT_DIR,
                "test-defaults",
                "--out-format",
                "magicc-input-files-point-end-of-year",
                "--regexp",
                ".*lai.*",
                "--drs",
                "CMIP6Output",
            ],
        )
    assert result.exit_code == 0

    assert (
        "ERROR:netcdf_scm:I don't know which MAGICC variable to use for input `lai`"
        in result.output
    )


def test_wrangling_blend_models(tmpdir, caplog):
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
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "test",
            "--regexp",
            ".*lai.*",
            "-f",
            "--prefix",
            "test-prefix",
            "--drs",
            "CMIP6Output",
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
                "--prefix",
                "test-prefix",
                "--drs",
                "CMIP6Output",
            ],
        )
    assert result_skip.exit_code == 0

    skip_str_file = "Skipped (already exists, not overwriting) {}".format(
        join(
            OUTPUT_DIR,
            "CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r1i1p1f2/Lmon/lai/gr/v20180917/netcdf-scm_lai_Lmon_CNRM-CM6-1_historical_r1i1p1f2_gr_200001-201412.MAG",
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
                "--drs",
                "CMIP6Output",
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
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "test",
            "--flat",
            "--out-format",
            "tuningstrucs-blend-model",
        ],
    )
    assert result.exit_code != 0


def test_wrangling_drs_replication(tmpdir):
    INPUT_DIR = join(TEST_DATA_CMIP6_CRUNCH_OUTPUT, "CMIP6/CMIP/CNRM-CERFACS")
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [INPUT_DIR, OUTPUT_DIR, "test", "--regexp", ".*lai.*", "--drs", "CMIP6Output"],
    )
    assert result.exit_code == 0
    assert isdir(join(OUTPUT_DIR, "CMIP6/CMIP/CNRM-CERFACS"))
