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
    result = runner.invoke(wrangle_openscm_csvs, [INPUT_DIR, OUTPUT_DIR, "--flat"])
    assert result.exit_code == 0

    assert "NetCDF SCM version: {}".format(netcdf_scm.__version__) in result.output

    assert "everything" in result.output

    assert len(listdir(OUTPUT_DIR)) == 12
