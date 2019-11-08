from os.path import join

import pytest
from click.testing import CliRunner

from netcdf_scm.cli import stitch_netcdf_scm_ncs


@pytest.mark.parametrize(
    "out_format", (["mag-files", "mag-files-average-year-mid-year"])
)
def test_stitching_results(
    tmpdir,
    update_expected_files,
    test_cmip6_crunch_output,
    test_cmip6_stitch_output,
    run_wrangling_comparison,
    out_format,
):
    input_dir = test_cmip6_crunch_output
    output_dir = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        stitch_netcdf_scm_ncs,
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
            # have to avoid files which will fail stiching and averaging
            "--regexp",
            "^(?!.*(piControl|thetao|CNRM-ESM2-1/ssp534-over|IPSL-CM6A-LR/.*/r1i1p1f1/Lmon/cSoilFast|IPSL-CM6A-LR/.*/r1i1p1f1/Lmon/gpp|CESM2|IPSL-CM6A-LR)).*$",
        ],
    )
    assert result.exit_code == 0, result.output

    run_wrangling_comparison(
        join(output_dir, "CMIP6"),
        join(test_cmip6_stitch_output, out_format),
        update=update_expected_files,
    )


@pytest.mark.parametrize(
    "out_format",
    (
        [
            "mag-files",
            "magicc-input-files-average-year-start-year",
            "magicc-input-files-point-start-year",
        ]
    ),
)
def test_stiching_and_normalise_results(
    tmpdir,
    update_expected_files,
    test_cmip6_crunch_output,
    test_cmip6_stitch_output,
    run_wrangling_comparison,
    out_format,
):
    input_dir = test_cmip6_crunch_output
    output_dir = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        stitch_netcdf_scm_ncs,
        [
            input_dir,
            output_dir,
            "cmip6output stitching with normalisation regression test",
            "--drs",
            "CMIP6Output",
            "-f",
            "--number-workers",
            1,
            "--out-format",
            out_format,
            # have to avoid files which will fail normalising
            "--regexp",
            "^(?!.*(piControl|CNRM-CM6-1/hist-aer|GFDL-CM4/1pctCO2|GFDL-CM4/abrupt-4xCO2|CESM2/historical|IPSL-CM6A-LR/historical|GISS-E2-1-G/abrupt-4xCO2)).*/tas/.*$",
            "--normalise",
            "31-yr-mean-after-branch-time",
        ],
    )
    assert result.exit_code == 0, result.output

    run_wrangling_comparison(
        join(output_dir, "CMIP6"),
        join(test_cmip6_stitch_output, "normalised", out_format),
        update=update_expected_files,
    )
