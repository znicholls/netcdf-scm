import datetime as dt
import os
from glob import glob
from os.path import isdir, join

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from pymagicc.io import MAGICCData

import netcdf_scm
from netcdf_scm.cli import wrangle_netcdf_scm_ncs
from netcdf_scm.iris_cube_wrappers import SCMCube


def test_wrangling_unsupported_format(tmpdir, caplog, test_cmip6_crunch_output):
    input_dir = test_cmip6_crunch_output
    output_dir = str(join(tmpdir, "new-sub-dir"))

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [input_dir, output_dir, "test-invalid-format", "--out-format", "junk"],
        )
    assert result.exit_code != 0
    assert "invalid choice: junk" in result.output
    assert isinstance(result.exception, SystemExit)


def test_wrangling_defaults(tmpdir, caplog, test_cmip6_crunch_output):
    input_dir = test_cmip6_crunch_output
    output_dir = str(join(tmpdir, "new-sub-dir"))
    test_wrangler = "test-defaults"

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                test_wrangler,
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
                # avoid thetao which we can't wrangle yet
                "--regexp",
                "^((?!thetao).)*$",
            ],
        )
    assert result.exit_code == 0

    assert "netcdf-scm: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory: {}".format(output_dir) in result.output

    assert "rlut" in result.output
    assert "lai" in result.output
    assert "cSoilFast" in result.output

    assert isdir(
        join(
            output_dir,
            "CMIP6/CMIP/BCC/BCC-CSM2-MR/1pctCO2/r1i1p1f1/Amon/rlut/gn/v20181015",
        )
    )
    assert isdir(
        join(
            output_dir,
            "CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp126/r1i1p1f1/Lmon/cSoilFast/gr/v20190121",
        )
    )
    assert isdir(join(output_dir, "CMIP6/CMIP/CNRM-CERFACS"))

    with open(
        join(
            output_dir,
            "CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r1i1p1f2/Lmon/lai/gr/v20180917/netcdf-scm_lai_Lmon_CNRM-CM6-1_historical_r1i1p1f2_gr_200201-200512.MAG",
        )
    ) as f:
        content = f.read()

    assert "Contact: {}".format(test_wrangler) in content

    with open(
        join(
            output_dir,
            "flat/netcdf-scm_lai_Lmon_CNRM-CM6-1_historical_r1i1p1f2_gr_200201-200512.MAG",
        )
    ) as f:
        content = f.read()

    assert "Making symlink to {}".format(join(output_dir, "flat")) in result.output
    assert "Contact: {}".format(test_wrangler) in content


def test_wrangling_magicc_input_files(tmpdir, caplog, test_marble_cmip5_crunch_output):
    input_dir = test_marble_cmip5_crunch_output
    output_dir = str(join(tmpdir, "new-sub-dir"))
    test_wrangler = "test wrangling magicc input point end of year files <email>"
    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                test_wrangler,
                "--out-format",
                "magicc-input-files-point-end-year",
                "--regexp",
                "^((?!historical).)*tas.*$",
                "--drs",
                "MarbleCMIP5",
                "--number-workers",
                1,
            ],
        )
    assert result.exit_code == 0

    assert "netcdf-scm: {}".format(netcdf_scm.__version__) in result.output
    assert "Making output directory: {}".format(output_dir) in result.output

    assert isdir(join(output_dir, "cmip5/1pctCO2/Amon/tas/CanESM2/r1i1p1"))
    assert isdir(join(output_dir, "cmip5/rcp26/Amon/tas/bcc-csm1-1/r1i1p1"))
    assert isdir(join(output_dir, "cmip5/rcp45/Amon/tas/HadCM3/r1i1p1"))
    assert isdir(join(output_dir, "cmip5/rcp45/Amon/tas/ACCESS1-0/r1i1p1"))
    assert isdir(join(output_dir, "cmip5/rcp85/Amon/tas/NorESM1-ME/r1i1p1"))

    with open(
        join(
            output_dir,
            "cmip5/rcp45/Amon/tas/HadCM3/r1i1p1/TAS_RCP45_HADCM3_R1I1P1_2006-2035_GLOBAL_SURFACE_TEMP.IN",
        )
    ) as f:
        content = f.read()

    assert "Contact: {}".format(test_wrangler) in content
    assert "timeseriestype: POINT_END_YEAR" in content
    assert "2006" in content
    assert "2006." not in content
    assert "2030" in content
    assert "2030." not in content
    assert "2035" in content
    assert "2035." not in content
    assert "2036" not in content
    assert "2036." not in content


def test_wrangling_magicc_input_files_error(tmpdir, caplog, test_cmip6_crunch_output):
    input_dir = test_cmip6_crunch_output
    output_dir = str(join(tmpdir, "new-sub-dir"))

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test-defaults",
                "--out-format",
                "magicc-input-files-point-end-year",
                "--regexp",
                ".*lai.*",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
            ],
        )
    assert result.exit_code != 0

    assert (
        str(
            "KeyError: "
            + '"'
            + "I don't know which MAGICC variable to use for input `lai`"
            + '"'
        )
        in result.output
    )


def test_wrangling_blend_models(tmpdir, caplog, test_cmip6_crunch_output):
    input_dir = test_cmip6_crunch_output
    output_dir = str(tmpdir)

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--out-format",
                "tuningstrucs-blend-model",
                # avoid thetao which we can't wrangle yet
                "--regexp",
                "^((?!thetao).)*$",
            ],
        )
    assert result.exit_code == 0, result.output

    assert "netcdf-scm: {}".format(netcdf_scm.__version__) in result.output

    assert "Wrangling re.compile" in result.output
    assert ".*" in result.output
    assert ".mat" in result.output

    assert len(glob(join(output_dir, "*.mat"))) == 123


def test_wrangling_handles_integer_units(tmpdir, caplog, test_cmip6_crunch_output):
    input_dir = test_cmip6_crunch_output
    output_dir = str(tmpdir)

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
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


@pytest.mark.parametrize("out_format", ["mag-files", "tuningstrucs-blend-model"])
def test_wrangling_force(tmpdir, caplog, out_format, test_cmip6_crunch_output):
    input_dir = test_cmip6_crunch_output
    output_dir = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [
            input_dir,
            output_dir,
            "test",
            "--regexp",
            ".*lai.*",
            "-f",
            "--prefix",
            "test-prefix",
            "--drs",
            "CMIP6Output",
            "--number-workers",
            1,
            "--out-format",
            out_format,
        ],
    )
    assert result.exit_code == 0

    caplog.clear()
    with caplog.at_level("INFO"):
        result_skip = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--regexp",
                ".*lai.*",
                "--prefix",
                "test-prefix",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
                "--out-format",
                out_format,
            ],
        )
    assert result_skip.exit_code == 0

    if out_format == "mag-files":
        expected_file = join(
            output_dir,
            "CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r1i1p1f2/Lmon/lai/gr/v20180917/netcdf-scm_lai_Lmon_CNRM-CM6-1_historical_r1i1p1f2_gr_200201-200512.MAG",
        )
    else:
        expected_file = join(
            output_dir,
            "test-prefix_LAI_HISTORICAL_R1I1P1F2_WORLD_SOUTHERN_HEMISPHERE_LAND.mat",
        )
    skip_str_file = "Skipped (already exists, not overwriting) {}".format(expected_file)
    assert skip_str_file in result_skip.output

    caplog.clear()
    with caplog.at_level("INFO"):
        result_force = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--regexp",
                ".*lai.*",
                "-f",
                "--prefix",
                "test-prefix",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
                "--out-format",
                out_format,
            ],
        )
    assert result_force.exit_code == 0
    assert skip_str_file not in result_force.output


def test_wrangling_blended_models_default_drs_error(tmpdir, test_cmip6_crunch_output):
    input_dir = test_cmip6_crunch_output
    output_dir = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [input_dir, output_dir, "test", "--out-format", "tuningstrucs-blend-model"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
    assert str(result.exception) == (
        "`drs` == 'None' is not supported yet. Please raise an issue at "
        "github.com/znicholls/netcdf-scm/ with your use case if you need this feature."
    )


def test_wrangling_drs_replication(tmpdir, test_cmip6_crunch_output):
    input_dir = join(test_cmip6_crunch_output, "CMIP/CNRM-CERFACS")
    output_dir = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [
            input_dir,
            output_dir,
            "test",
            "--regexp",
            ".*lai.*",
            "--drs",
            "CMIP6Output",
            "--number-workers",
            1,
        ],
    )
    assert result.exit_code == 0
    assert isdir(join(output_dir, "CMIP6/CMIP/CNRM-CERFACS"))


def test_wrangling_annual_mean_file(tmpdir, test_data_root_dir):
    input_dir = join(
        test_data_root_dir,
        "marble-cmip5-annual-output/cmip5/rcp26/Amon/tas/bcc-csm1-1/r1i1p1",
    )
    output_dir = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [input_dir, output_dir, "test", "--drs", "MarbleCMIP5", "--number-workers", 1],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert (
        "ValueError: Please raise an issue at "
        "github.com/znicholls/netcdf-scm/issues to discuss how to handle "
        "non-monthly data wrangling"
    ) in result.output


@pytest.mark.parametrize(
    "target_unit,conv_factor", (["kg / m**2 / yr", 3.155695e07], ["g / m**2 / s", 1e03])
)
def test_wrangling_units_specs(
    tmpdir, test_cmip6_crunch_output, target_unit, conv_factor, caplog
):
    target_units = pd.DataFrame(
        [["fgco2", target_unit], ["tos", "K"]], columns=["variable", "unit"]
    )
    target_units_csv = join(tmpdir, "target_units.csv")
    target_units.to_csv(target_units_csv, index=False)

    runner = CliRunner()

    input_dir = join(test_cmip6_crunch_output, "CMIP/CCCma")
    output_dir = str(tmpdir)

    caplog.clear()
    with caplog.at_level("INFO"):
        result_raw = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
            ],
        )
    assert result_raw.exit_code == 0
    assert "Converting units" not in caplog.messages

    expected_file = join(
        output_dir,
        "CMIP6/CMIP/CCCma/CanESM5/piControl/r1i1p1f1/Omon/fgco2/gn/v20190429/netcdf-scm_fgco2_Omon_CanESM5_piControl_r1i1p1f1_gn_600101-600103.MAG",
    )

    res_raw = MAGICCData(expected_file)
    caplog.clear()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
                "--target-units-specs",
                target_units_csv,
                "--force",
            ],
        )
    assert result.exit_code == 0
    assert (
        "Converting units of fgco2 from kg m^-2 s^-1 to {}".format(target_unit)
        in caplog.messages
    )

    res = MAGICCData(expected_file)

    np.testing.assert_allclose(
        res_raw.timeseries() * conv_factor, res.timeseries(), rtol=1e-5
    )


def test_wrangling_units_specs_area_sum(tmpdir, test_cmip6_crunch_output, caplog):
    target_unit = "Gt / yr"
    target_units = pd.DataFrame(
        [["fgco2", target_unit], ["tos", "K"]], columns=["variable", "unit"]
    )
    target_units_csv = join(tmpdir, "target_units.csv")
    target_units.to_csv(target_units_csv, index=False)

    runner = CliRunner()

    input_dir = join(test_cmip6_crunch_output, "CMIP/CCCma")
    output_dir = str(tmpdir)

    result_raw = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [input_dir, output_dir, "test", "--drs", "CMIP6Output", "--number-workers", 1],
    )

    expected_file = join(
        output_dir,
        "CMIP6/CMIP/CCCma/CanESM5/piControl/r1i1p1f1/Omon/fgco2/gn/v20190429/netcdf-scm_fgco2_Omon_CanESM5_piControl_r1i1p1f1_gn_600101-600103.MAG",
    )
    assert result_raw.exit_code == 0
    res_raw = MAGICCData(expected_file)

    caplog.clear()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
                "--target-units-specs",
                target_units_csv,
                "--force",
            ],
        )

    assert result.exit_code == 0
    assert (
        "Converting units of fgco2 from kg m^-2 s^-1 to {}".format(target_unit)
        in caplog.messages
    )
    res = MAGICCData(expected_file)

    assert sorted(res["region"].tolist()) == sorted(res_raw["region"].tolist())
    for region, df in res.timeseries().groupby("region"):
        for k, v in res.metadata.items():
            if "{} (".format(SCMCube._convert_region_to_area_key(region)) in k:
                unit = k.split("(")[-1].split(")")[0]
                assert unit == "m**2", "assumed unit for test has changed..."
                conv_factor = (
                    float(v) * 10 ** -12 * 3.155695e07
                )  # area x mass conv x time conv
                break

        np.testing.assert_allclose(
            df.values, res_raw.filter(region=region).values * conv_factor, rtol=1e-5
        )


def test_wrangling_mag_file(tmpdir, test_cmip6_crunch_output, caplog):
    runner = CliRunner()

    input_dir = join(test_cmip6_crunch_output, "ScenarioMIP/IPSL/IPSL-CM6A-LR")
    output_dir = str(tmpdir)

    caplog.clear()
    with caplog.at_level("INFO"):
        result_raw = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
            ],
        )
    assert result_raw.exit_code == 0

    expected_file = join(
        output_dir,
        "CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp126/r1i1p1f1/Lmon/cSoilFast/gr/v20190121/netcdf-scm_cSoilFast_Lmon_IPSL-CM6A-LR_ssp126_r1i1p1f1_gr_202501-204012.MAG",
    )

    with open(expected_file) as f:
        content = f.read()

    assert "THISFILE_TIMESERIESTYPE = 'MONTHLY'" in content


def _get_expected_wrangled_ts(res_raw, out_format_mag):
    if out_format_mag.endswith("average-year-start-year"):
        # drop out last year as we don't want the wrangler to add an extra year to the
        # data
        return res_raw.time_mean("AS").timeseries().iloc[:, :-1]

    if out_format_mag.endswith("average-year-mid-year"):
        # drop out last year as we don't want the wrangler to add an extra year to the
        # data
        return res_raw.time_mean("AC").timeseries()

    if out_format_mag.endswith("average-year-end-year"):
        # drop out first year as we don't want the wrangler to add an extra year to the
        # data
        return res_raw.time_mean("A").timeseries().iloc[:, 1:]

    if out_format_mag.endswith("point-start-year"):
        # drop out last year as we don't want the wrangler to add an extra year to the
        # data
        return res_raw.resample("AS").timeseries().iloc[:, :-1]

    if out_format_mag.endswith("point-mid-year"):
        # drop out last year as we don't want the wrangler to add an extra year to the
        # data
        out_time_points = [
            dt.datetime(y, 7, 1)
            for y in range(res_raw["time"].min().year, res_raw["time"].max().year + 1,)
        ]
        return res_raw.interpolate(target_times=out_time_points).timeseries()

    if out_format_mag.endswith("point-end-year"):
        # drop out first year as we don't want the wrangler to add an extra year to the
        # data
        return res_raw.resample("A").timeseries().iloc[:, 1:]

    raise AssertionError("shouldn't get here")


@pytest.mark.parametrize(
    "out_format_mag",
    (
        "mag-files-average-year-start-year",
        "mag-files-average-year-mid-year",
        "mag-files-average-year-end-year",
        "mag-files-point-start-year",
        "mag-files-point-mid-year",
        "mag-files-point-end-year",
    ),
)
def test_wrangling_mag_file_operations(
    tmpdir, test_cmip6_crunch_output, caplog, out_format_mag
):
    runner = CliRunner()

    input_dir = join(test_cmip6_crunch_output, "ScenarioMIP/IPSL/IPSL-CM6A-LR")
    output_dir = str(tmpdir)

    caplog.clear()
    with caplog.at_level("INFO"):
        result_raw = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
            ],
        )
    assert result_raw.exit_code == 0

    expected_file_raw = join(
        output_dir,
        "CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp126/r1i1p1f1/Lmon/cSoilFast/gr/v20190121/netcdf-scm_cSoilFast_Lmon_IPSL-CM6A-LR_ssp126_r1i1p1f1_gr_202501-204012.MAG",
    )

    res_raw = MAGICCData(expected_file_raw)
    res_raw_resampled = _get_expected_wrangled_ts(res_raw, out_format_mag)

    caplog.clear()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--out-format",
                out_format_mag,
                "--number-workers",
                1,
            ],
        )
    assert result.exit_code == 0

    expected_file = join(
        output_dir,
        "CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp126/r1i1p1f1/Lmon/cSoilFast/gr/v20190121/netcdf-scm_cSoilFast_Lmon_IPSL-CM6A-LR_ssp126_r1i1p1f1_gr_2025-2040.MAG",
    )

    res = MAGICCData(expected_file)

    np.testing.assert_allclose(res_raw_resampled, res.timeseries(), rtol=2 * 1e-3)
    with open(expected_file) as f:
        content = f.read()

    assert (
        "THISFILE_TIMESERIESTYPE = '{}'".format(
            out_format_mag.replace("mag-files-", "").replace("-", "_").upper()
        )
        in content
    )


def test_wrangling_in_file(tmpdir, test_cmip6_crunch_output, caplog):
    runner = CliRunner()

    input_dir = join(test_cmip6_crunch_output, "CMIP/IPSL/IPSL-CM6A-LR/historical")
    output_dir = str(tmpdir)

    caplog.clear()
    with caplog.at_level("INFO"):
        result_raw = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
                "--out-format",
                "magicc-input-files",
                "--regexp",
                ".*tas.*",
            ],
        )
    assert result_raw.exit_code == 0, result_raw.stdout

    # also a global file but don't worry about that
    expected_file = join(
        output_dir,
        "CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/tas/gr/v20180803/TAS_HISTORICAL_IPSL-CM6A-LR_R1I1P1F1_191001-191003_FOURBOX_SURFACE_TEMP.IN",
    )

    with open(expected_file) as f:
        content = f.read()

    assert "timeseriestype: MONTHLY" in content


@pytest.mark.parametrize(
    "out_format_in_file",
    (
        "magicc-input-files-average-year-start-year",
        "magicc-input-files-average-year-mid-year",
        "magicc-input-files-average-year-end-year",
        "magicc-input-files-point-start-year",
        "magicc-input-files-point-mid-year",
        "magicc-input-files-point-end-year",
    ),
)
def test_wrangling_in_file_operations(
    tmpdir, test_cmip6_crunch_output, caplog, out_format_in_file
):
    runner = CliRunner()

    input_dir = join(test_cmip6_crunch_output, "CMIP/IPSL/IPSL-CM6A-LR/piControl")
    output_dir = str(tmpdir)

    caplog.clear()
    with caplog.at_level("INFO"):
        result_raw = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
                "--out-format",
                "magicc-input-files",
                "--regexp",
                ".*tas.*",
            ],
        )
    assert result_raw.exit_code == 0, result_raw.stdout

    # also a global file but don't worry about that
    expected_file_raw = join(
        output_dir,
        "CMIP6/CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Amon/tas/gr/v20181123/TAS_PICONTROL_IPSL-CM6A-LR_R1I1P1F1_284001-285912_FOURBOX_SURFACE_TEMP.IN",
    )

    res_raw = MAGICCData(expected_file_raw)
    res_raw_resampled = _get_expected_wrangled_ts(res_raw, out_format_in_file)

    caplog.clear()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--out-format",
                out_format_in_file,
                "--regexp",
                ".*tas.*",
                "--number-workers",
                1,
            ],
        )
    assert result.exit_code == 0, result.stdout

    os.listdir(
        join(
            output_dir,
            "CMIP6/CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Amon/tas/gr/v20181123/",
        )
    )
    expected_file = join(
        output_dir,
        "CMIP6/CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Amon/tas/gr/v20181123/TAS_PICONTROL_IPSL-CM6A-LR_R1I1P1F1_2840-2859_FOURBOX_SURFACE_TEMP.IN",
    )

    res = MAGICCData(expected_file)

    np.testing.assert_allclose(res_raw_resampled, res.timeseries(), rtol=2 * 1e-3)
    with open(expected_file) as f:
        content = f.read()

    assert (
        "timeseriestype: {}".format(
            out_format_in_file.replace("magicc-input-files-", "")
            .replace("-", "_")
            .upper()
        )
        in content
    )


@pytest.mark.xfail(
    message="Some of these errors are still cryptic because scmdata isn't inteprolating as intended"
)
@pytest.mark.parametrize(
    "out_format",
    (
        "mag-files-average-year-start-year",
        "mag-files-average-year-mid-year",
        "mag-files-average-year-end-year",
        "mag-files-point-start-year",
        "mag-files-point-mid-year",
        "mag-files-point-end-year",
        "magicc-input-files-average-year-start-year",
        "magicc-input-files-average-year-mid-year",
        "magicc-input-files-average-year-end-year",
        "magicc-input-files-point-start-year",
        "magicc-input-files-point-mid-year",
        "magicc-input-files-point-end-year",
    ),
)
def test_wrangling_with_operation_single_timepoint(
    tmpdir, test_cmip6_crunch_output, caplog, out_format
):
    runner = CliRunner()

    input_dir = join(
        test_cmip6_crunch_output, "CMIP/NCAR/CESM2/historical/r7i1p1f1/Amon/tas"
    )
    output_dir = str(tmpdir)

    caplog.clear()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                input_dir,
                output_dir,
                "test",
                "--drs",
                "CMIP6Output",
                "--number-workers",
                1,
                "--out-format",
                out_format,
            ],
        )
    assert result.exit_code != 0
    assert (
        "ValueError: We cannot yet write `{}` if the output data will have only one timestep".format(
            out_format
        )
        in result.stdout
    )
