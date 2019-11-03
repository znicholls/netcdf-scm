from os.path import join

import pytest
from click.testing import CliRunner

from netcdf_scm.cli import wrangle_netcdf_scm_ncs


@pytest.mark.parametrize(
    "out_format", (["mag-files", "mag-files-average-year-mid-year"])
)
def test_wrangling_results(
    tmpdir,
    update_expected_files,
    test_cmip6_crunch_output,
    test_cmip6_wrangle_output,
    run_wrangling_comparison,
    out_format,
):
    input_dir = test_cmip6_crunch_output
    output_dir = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [
            input_dir,
            output_dir,
            "cmip6output wrangling regression test",
            "--drs",
            "CMIP6Output",
            "-f",
            "--number-workers",
            1,
            "--out-format",
            out_format,
            # have to avoid files which only have a single year and will fail with
            # average wrangling
            "--regexp",
            "^(?!.*(CESM2/historical/r7i1p1f1|IPSL-CM6A-LR/historical/r1i1p1f1)).*(/tas/|/tos/|/rlut/|/lai/).*$",
        ],
    )
    assert result.exit_code == 0, result.output

    run_wrangling_comparison(
        join(output_dir, "CMIP6"),
        join(test_cmip6_wrangle_output, out_format),
        update=update_expected_files,
    )
