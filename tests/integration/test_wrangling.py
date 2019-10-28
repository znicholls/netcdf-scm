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
    INPUT_DIR = test_cmip6_crunch_output
    OUTPUT_DIR = str(join(tmpdir, "new-sub-dir"))

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [INPUT_DIR, OUTPUT_DIR, "test-invalid-format", "--out-format", "junk"],
        )
    assert result.exit_code != 0
    assert "invalid choice: junk" in result.output
    assert isinstance(result.exception, SystemExit)


def test_wrangling_defaults(tmpdir, caplog, test_cmip6_crunch_output):
    INPUT_DIR = test_cmip6_crunch_output
    OUTPUT_DIR = str(join(tmpdir, "new-sub-dir"))
    test_wrangler = "test-defaults"

    runner = CliRunner()
    with caplog.at_level("INFO"):
        result = runner.invoke(
            wrangle_netcdf_scm_ncs,
            [
                INPUT_DIR,
                OUTPUT_DIR,
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
            "CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r1i1p1f2/Lmon/lai/gr/v20180917/netcdf-scm_lai_Lmon_CNRM-CM6-1_historical_r1i1p1f2_gr_200201-200512.MAG",
        )
    ) as f:
        content = f.read()

    assert "Contact: {}".format(test_wrangler) in content

    with open(
        join(
            OUTPUT_DIR,
            "flat/netcdf-scm_lai_Lmon_CNRM-CM6-1_historical_r1i1p1f2_gr_200201-200512.MAG",
        )
    ) as f:
        content = f.read()

    assert "Making symlink to {}".format(join(OUTPUT_DIR, "flat")) in result.output
    assert "Contact: {}".format(test_wrangler) in content


def test_wrangling_magicc_input_files(tmpdir, caplog, test_marble_cmip5_crunch_output):
    INPUT_DIR = test_marble_cmip5_crunch_output
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
                "^((?!historical).)*tas.*$",
                "--drs",
                "MarbleCMIP5",
                "--number-workers",
                1,
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


def test_wrangling_magicc_input_files_error(tmpdir, caplog, test_cmip6_crunch_output):
    INPUT_DIR = test_cmip6_crunch_output
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
                "--number-workers",
                1,
            ],
        )
    assert result.exit_code == 0

    assert (
        "ERROR:netcdf_scm:I don't know which MAGICC variable to use for input `lai`"
        in result.output
    )


def test_wrangling_blend_models(tmpdir, caplog, test_cmip6_crunch_output):
    INPUT_DIR = test_cmip6_crunch_output
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

    assert len(glob(join(OUTPUT_DIR, "*.mat"))) == 123


def test_wrangling_handles_integer_units(tmpdir, caplog, test_cmip6_crunch_output):
    INPUT_DIR = test_cmip6_crunch_output
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


@pytest.mark.parametrize("out_format", ["mag-files", "tuningstrucs-blend-model"])
def test_wrangling_force(tmpdir, caplog, out_format, test_cmip6_crunch_output):
    INPUT_DIR = test_cmip6_crunch_output
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
                INPUT_DIR,
                OUTPUT_DIR,
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
            OUTPUT_DIR,
            "CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/r1i1p1f2/Lmon/lai/gr/v20180917/netcdf-scm_lai_Lmon_CNRM-CM6-1_historical_r1i1p1f2_gr_200201-200512.MAG",
        )
    else:
        expected_file = join(
            OUTPUT_DIR,
            "test-prefix_LAI_HISTORICAL_R1I1P1F2_WORLD_SOUTHERN_HEMISPHERE_LAND.mat",
        )
    skip_str_file = "Skipped (already exists, not overwriting) {}".format(expected_file)
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
                "--number-workers",
                1,
                "--out-format",
                out_format,
            ],
        )
    assert result_force.exit_code == 0
    assert skip_str_file not in result_force.output


def test_wrangling_blended_models_default_drs_error(tmpdir, test_cmip6_crunch_output):
    INPUT_DIR = test_cmip6_crunch_output
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [INPUT_DIR, OUTPUT_DIR, "test", "--out-format", "tuningstrucs-blend-model"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
    assert str(result.exception) == (
        "`drs` == 'None' is not supported yet. Please raise an issue at "
        "github.com/znicholls/netcdf-scm/ with your use case if you need this feature."
    )


def test_wrangling_drs_replication(tmpdir, test_cmip6_crunch_output):
    INPUT_DIR = join(test_cmip6_crunch_output, "CMIP/CNRM-CERFACS")
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
            "--drs",
            "CMIP6Output",
            "--number-workers",
            1,
        ],
    )
    assert result.exit_code == 0
    assert isdir(join(OUTPUT_DIR, "CMIP6/CMIP/CNRM-CERFACS"))


def test_wrangling_annual_mean_file(tmpdir, test_data_root_dir):
    INPUT_DIR = join(
        test_data_root_dir,
        "marble-cmip5-annual-output/cmip5/rcp26/Amon/tas/bcc-csm1-1/r1i1p1",
    )
    OUTPUT_DIR = str(tmpdir)

    runner = CliRunner()
    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [INPUT_DIR, OUTPUT_DIR, "test", "--drs", "MarbleCMIP5", "--number-workers", 1],
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
    tmpdir, test_cmip6_crunch_output, target_unit, conv_factor
):
    target_units = pd.DataFrame(
        [["fgco2", target_unit], ["tos", "K"]], columns=["variable", "unit"]
    )
    target_units_csv = join(tmpdir, "target_units.csv")
    target_units.to_csv(target_units_csv, index=False)

    runner = CliRunner()

    INPUT_DIR = join(test_cmip6_crunch_output, "CMIP/CCCma")
    OUTPUT_DIR = str(tmpdir)

    result_raw = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [INPUT_DIR, OUTPUT_DIR, "test", "--drs", "CMIP6Output", "--number-workers", 1],
    )

    expected_file = join(
        OUTPUT_DIR,
        "CMIP6/CMIP/CCCma/CanESM5/piControl/r1i1p1f1/Omon/fgco2/gn/v20190429/netcdf-scm_fgco2_Omon_CanESM5_piControl_r1i1p1f1_gn_600101-600103.MAG",
    )

    res_raw = MAGICCData(expected_file)

    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
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
    res = MAGICCData(expected_file)

    np.testing.assert_allclose(
        res_raw.timeseries() * conv_factor, res.timeseries(), rtol=1e-5
    )


def test_wrangling_units_specs_area_sum(tmpdir, test_cmip6_crunch_output):
    target_unit = "Gt / yr"
    target_units = pd.DataFrame(
        [["fgco2", target_unit], ["tos", "K"]], columns=["variable", "unit"]
    )
    target_units_csv = join(tmpdir, "target_units.csv")
    target_units.to_csv(target_units_csv, index=False)

    runner = CliRunner()

    INPUT_DIR = join(test_cmip6_crunch_output, "CMIP/CCCma")
    OUTPUT_DIR = str(tmpdir)

    result_raw = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [INPUT_DIR, OUTPUT_DIR, "test", "--drs", "CMIP6Output", "--number-workers", 1],
    )

    expected_file = join(
        OUTPUT_DIR,
        "CMIP6/CMIP/CCCma/CanESM5/piControl/r1i1p1f1/Omon/fgco2/gn/v20190429/netcdf-scm_fgco2_Omon_CanESM5_piControl_r1i1p1f1_gn_600101-600103.MAG",
    )

    res_raw = MAGICCData(expected_file)

    result = runner.invoke(
        wrangle_netcdf_scm_ncs,
        [
            INPUT_DIR,
            OUTPUT_DIR,
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
