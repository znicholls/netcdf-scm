from os.path import join

import pytest
from click.testing import CliRunner

from netcdf_scm.cli import stich_netcdf_scm_ncs


@pytest.mark.parametrize(
    "out_format", ([
        "mag-files",
        "mag-files-average-year-mid-year",
        "magicc-input-files-average-year-start-year",
    ])
)
def test_stitching_results(
    tmpdir,
    update_expected_files,
    test_cmip6_crunch_output,
    test_cmip6_stich_output,
    run_stitching_comparison,
    out_format,
):
    input_dir = test_cmip6_crunch_output
    output_dir = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        stich_netcdf_scm_ncs,
        [
            input_dir,
            output_dir,
            "cmip6output stitching regression test",
            "--drs",
            "CMIP6Output",
            "-f",
            "--number-workers",
            1,
            "--out-format",
            out_format,
            # have to avoid files which only have a single year and will fail with
            # average stitching
            "--regexp",
            "^(?!.*(CESM2/historical/r7i1p1f1|IPSL-CM6A-LR/historical/r1i1p1f1)).*(/tas/|/tos/|/rlut/|/lai/).*$",
        ],
    )
    assert result.exit_code == 0, result.output

    run_stitching_comparison(
        join(output_dir, "CMIP6"),
        join(test_cmip6_stich_output, out_format),
        update=update_expected_files,
    )


@pytest.mark.parametrize(
    "out_format", ([
        "mag-files",
        "mag-files-average-year-mid-year",
        "magicc-input-files-average-year-start-year",
    ])
)
def test_stitching_results_with_normalise(
    tmpdir,
    update_expected_files,
    test_cmip6_crunch_output,
    test_cmip6_stich_and_normalise_output,
    run_stitching_comparison,
    out_format,
):
    input_dir = test_cmip6_crunch_output
    output_dir = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        stich_netcdf_scm_ncs,
        [
            input_dir,
            output_dir,
            "cmip6output stitching regression test with normalisation",
            "--drs",
            "CMIP6Output",
            "-f",
            "--number-workers",
            1,
            "--out-format",
            out_format,
            # have to avoid files which only have a single year and will fail with
            # average stitching
            "--regexp",
            "^(?!.*(CESM2/historical/r7i1p1f1|IPSL-CM6A-LR/historical/r1i1p1f1)).*tas.*$",
            "--normalise",
        ],
    )
    assert result.exit_code == 0, result.output

    run_stitching_comparison(
        join(output_dir, "CMIP6"),
        join(test_cmip6_stich_and_normalise_output, out_format),
        update=update_expected_files,
    )
