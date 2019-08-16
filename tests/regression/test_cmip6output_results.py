from os.path import join

from click.testing import CliRunner

from netcdf_scm.cli import crunch_data


def test_crunching(
    tmpdir,
    update_expected_files,
    test_data_cmip6output_dir,
    test_cmip6_crunch_output,
    run_crunching_comparison,
):
    INPUT_DIR = test_data_cmip6output_dir
    OUTPUT_DIR = str(tmpdir)
    REGIONS_TO_GET = [
        "World",
        "World|Northern Hemisphere",
        "World|Southern Hemisphere",
        "World|Land",
        "World|Ocean",
        "World|Northern Hemisphere|Land",
        "World|Southern Hemisphere|Land",
        "World|Northern Hemisphere|Ocean",
        "World|Southern Hemisphere|Ocean",
        "World|North Atlantic Ocean",
        "World|El Nino N3.4",
    ]

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
            1,
            "--regions",
            ",".join(REGIONS_TO_GET)
        ],
    )
    assert result.exit_code == 0, result.output
    run_crunching_comparison(
        join(OUTPUT_DIR, "netcdf-scm-crunched", "CMIP6"),
        test_cmip6_crunch_output,
        update=update_expected_files,
    )
