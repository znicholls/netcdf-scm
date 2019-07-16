from os.path import join

from click.testing import CliRunner
from conftest import (
    TEST_DATA_ROOT_DIR,
    TEST_DATA_CMIP6Output_DIR,
    run_crunching_comparison,
)

from netcdf_scm.cli import crunch_data

EXPECTED_FILES_DIR = join(TEST_DATA_ROOT_DIR, "expected-crunching-output")


def test_crunching(tmpdir, update_expected_files):
    INPUT_DIR = TEST_DATA_CMIP6Output_DIR
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        crunch_data,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "cmip6output crunching regression test",
            "--drs",
            "CMIP6Output",
            "-f",
            "--small-number-workers",
            2
        ],
    )
    assert result.exit_code == 0, result.output
    run_crunching_comparison(
        join(OUTPUT_DIR, "netcdf-scm-crunched", "CMIP6"),
        join(EXPECTED_FILES_DIR, "cmip6output", "CMIP6"),
        update=update_expected_files,
    )
