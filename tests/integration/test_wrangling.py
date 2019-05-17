from os import listdir
from os.path import isdir, join

from click.testing import CliRunner


import netcdf_scm
from netcdf_scm.cli import wrangle_openscm_csvs


from conftest import TEST_DATA_OPENSCMCSVS_DIR


def test_wrangling_defaults(tmpdir):
    INPUT_DIR = TEST_DATA_OPENSCMCSVS_DIR
    OUTPUT_DIR = str(join(tmpdir, "new-sub-dir"))

    runner = CliRunner()
    result = runner.invoke(wrangle_openscm_csvs, [INPUT_DIR, OUTPUT_DIR])
    assert result.exit_code == 0

    assert "NetCDF SCM version: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory:" in result.output

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


def test_wrangling_var(tmpdir):
    INPUT_DIR = TEST_DATA_OPENSCMCSVS_DIR
    OUTPUT_DIR = str(tmpdir)
    VAR_TO_WRANGLE = ".*rsut.*"

    runner = CliRunner()
    result = runner.invoke(
        wrangle_openscm_csvs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--var-to-wrangle",
            VAR_TO_WRANGLE,
            "--nested",
            "--out-format",
            "tuningstrucs",
        ],
    )
    assert result.exit_code == 0

    assert "NetCDF SCM version: {}".format(netcdf_scm.__version__) in result.output

    assert "rsut" in result.output
    assert "rlut" not in result.output

    assert isdir(
        join(OUTPUT_DIR, "cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/1pctCO2/r1i1p1f1/Amon/rsut")
    )
    assert isdir(
        join(
            OUTPUT_DIR,
            "cmip6/CMIP6/ScenarioMIP/MRI/MRI-ESM2-0/ssp126/r1i1p1f1/Amon/rsut",
        )
    )


def test_wrangling_flat(tmpdir):
    INPUT_DIR = TEST_DATA_OPENSCMCSVS_DIR
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_openscm_csvs, [INPUT_DIR, OUTPUT_DIR, "--flat", "--drs", "CMIP6Output"]
    )
    assert result.exit_code == 0

    assert "NetCDF SCM version: {}".format(netcdf_scm.__version__) in result.output

    assert "Wrangling re.compile" in result.output
    assert ".*" in result.output
    assert ".mat" in result.output

    assert len(listdir(OUTPUT_DIR)) == 27
    assert (
        "ts_CMIP_1pctCO2_r1i1p1f1_unspecified_toa_outgoing_shortwave_flux_World.mat"
        not in result.output
    )


def test_wrangling_handles_integer_units(tmpdir):
    INPUT_DIR = TEST_DATA_OPENSCMCSVS_DIR
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_openscm_csvs, [INPUT_DIR, OUTPUT_DIR, "--var-to-wrangle", ".*lai.*"]
    )
    assert result.exit_code == 0

    assert "NetCDF SCM version: {}".format(netcdf_scm.__version__) in result.output

    assert "lai" in result.output


def test_wrangling_force(tmpdir):
    INPUT_DIR = TEST_DATA_OPENSCMCSVS_DIR
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_openscm_csvs,
        [INPUT_DIR, OUTPUT_DIR, "--var-to-wrangle", ".*lai.*", "-f"],
    )
    assert result.exit_code == 0

    result_skip = runner.invoke(
        wrangle_openscm_csvs, [INPUT_DIR, OUTPUT_DIR, "--var-to-wrangle", ".*lai.*"]
    )
    assert result_skip.exit_code == 0

    skip_str_header = (
        "Skipped (already exist, not overwriting)\n"
        "========================================\n"
    )
    assert skip_str_header in result_skip.output
    assert "-" in result_skip.output

    skip_str_file = "- {}".format(
        join(
            OUTPUT_DIR,
            "cmip6/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r2i1p1f2/Lmon/lai/gr/v20181126/ts_CMIP_historical_r2i1p1f2_unspecified_leaf_area_index_World_Southern_Hemisphere_Ocean.mat",
        )
    )
    assert skip_str_file in result_skip.output

    result_force = runner.invoke(
        wrangle_openscm_csvs,
        [INPUT_DIR, OUTPUT_DIR, "--var-to-wrangle", ".*lai.*", "-f"],
    )
    assert result_force.exit_code == 0
    assert skip_str_header in result_force.output
    assert skip_str_file not in result_force.output


def test_wrangling_force_flat(tmpdir):
    INPUT_DIR = TEST_DATA_OPENSCMCSVS_DIR
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_openscm_csvs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--var-to-wrangle",
            ".*lai.*",
            "-f",
            "--flat",
            "--drs",
            "CMIP6Output",
        ],
    )
    assert result.exit_code == 0

    result_skip = runner.invoke(
        wrangle_openscm_csvs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--var-to-wrangle",
            ".*lai.*",
            "--flat",
            "--drs",
            "CMIP6Output",
        ],
    )
    assert result_skip.exit_code == 0

    skip_str_header = (
        "Skipped (already exist, not overwriting)\n"
        "========================================\n"
    )
    assert skip_str_header in result_skip.output
    assert "-" in result_skip.output

    skip_str_file = "- {}".format(
        join(
            OUTPUT_DIR,
            "ts_CMIP_historical_r2i1p1f2_unspecified_leaf_area_index_World.mat",
        )
    )
    assert skip_str_file in result_skip.output

    result_force = runner.invoke(
        wrangle_openscm_csvs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--var-to-wrangle",
            ".*lai.*",
            "-f",
            "--flat",
            "--drs",
            "CMIP6Output",
        ],
    )
    assert result_force.exit_code == 0
    assert skip_str_header in result_force.output
    assert skip_str_file not in result_force.output


def test_wrangling_default_drs_error(tmpdir):
    INPUT_DIR = TEST_DATA_OPENSCMCSVS_DIR
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(wrangle_openscm_csvs, [INPUT_DIR, OUTPUT_DIR, "--flat"])
    assert result.exit_code != 0
