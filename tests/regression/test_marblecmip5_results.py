from os.path import join


from click.testing import CliRunner


from netcdf_scm.cli import crunch_data


from conftest import TEST_DATA_ROOT_DIR, TEST_DATA_MARBLE_CMIP5_DIR, run_crunching_comparison


EXPECTED_FILES_DIR = join(TEST_DATA_ROOT_DIR, "expected-crunching-output")


def test_crunching(tmpdir, update_expected_files):
    INPUT_DIR = TEST_DATA_MARBLE_CMIP5_DIR
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        crunch_data,
        [
            INPUT_DIR,
            OUTPUT_DIR,
            "--cube-type",
            "MarbleCMIP5",
            "-f",
        ],
    )
    assert result.exit_code == 0, result.output
    run_crunching_comparison(
        join(OUTPUT_DIR, "netcdf-scm-crunched", "cmip5"),
        join(EXPECTED_FILES_DIR, "marble-cmip5", "cmip5"),
        update=update_expected_files,
    )
