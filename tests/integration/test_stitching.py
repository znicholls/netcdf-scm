import glob
import logging
import os.path
import re

import numpy as np
import pytest
from click.testing import CliRunner
from pymagicc.io import MAGICCData

from netcdf_scm.cli import stitch_netcdf_scm_ncs
from netcdf_scm.io import load_scmdataframe

# TODO:
# - create small (i.e. regridded using cdo remapbil,n4 <infile> <outfile>) test files which don't clash with existing data and match all the tests
    # - test data (timeseries which can be stitched, one ssp plus hist, one hist plus piControl, one 1pctCO2 plus piControl, one BCC-CSM2-MR hist plus BCC-CSM2-MR piControl)
# - start passing tests
    # - tests of errors when parent isn't there
    # - tests of errors when branch time isn't in data
    # - tests of errors when not enough years in piControl are there
    # - tests of log messages when doing the BCC-CSM2-MR fix
    # - tests of normalisation or not


def _do_generic_stitched_data_tests(stiched_scmdf):
    assert stiched_scmdf["scenario"].nunique() == 1
    assert "(child) branch_time_in_parent"  in stiched_scmdf.metadata
    assert "(child) parent_experiment_id"  in stiched_scmdf.metadata
    assert "(parent) source_id"  in stiched_scmdf.metadata
    assert "(parent) experiment_id"  in stiched_scmdf.metadata


def _do_generic_normalised_data_tests(normalised_scmdf):
    assert normalised_scmdf["scenario"].nunique() == 1
    assert "(child) branch_time_in_parent"  in normalised_scmdf.metadata
    assert "(child) parent_experiment_id"  in normalised_scmdf.metadata
    assert "(normalisation) source_id"  in normalised_scmdf.metadata
    assert "(normalisation) experiment_id"  in normalised_scmdf.metadata
    assert "normalisation method" in normalised_scmdf.metadata


def test_stitching_default(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_default"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            stitch_netcdf_scm_ncs,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "--number-workers",
                1,
                "--regexp",
                ".*EC-Earth3-Veg.*ssp585.*r1i1p1f1.*hfds.*",
            ],
        )

    assert result.exit_code == 0, result.stderr

    out_files = glob.glob(os.path.join(output_dir, "flat", "*.MAG"))
    assert len(out_files) == 1

    res = MAGICCData(out_files[0])

    _do_generic_stitched_data_tests(res)

    child_path = res.metadata["(child) netcdf-scm crunched file"]
    assert "ssp585" in child_path
    assert "r1i1p1f1" in child_path
    assert "EC-Earth3-Veg" in child_path
    assert "hfds" in child_path

    child = load_scmdataframe(os.path.join(test_cmip6_crunch_output, child_path.replace("CMIP6/", "")))

    for region in ["World", "World|North Atlantic Ocean"]:
        for y in range(2015, 2017):
            np.testing.assert_allclose(
                res.filter(region=region, year=y).values,
                child.filter(region=region, year=y).values,
                rtol=1e-5,
            )

    parent_path = res.metadata["(parent) netcdf-scm crunched file"]
    assert "historical" in parent_path
    assert "r1i1p1f1" in parent_path
    assert "EC-Earth3-Veg" in parent_path
    assert "hfds" in parent_path

    parent = load_scmdataframe(os.path.join(test_cmip6_crunch_output, parent_path.replace("CMIP6/", "")))

    for region in ["World", "World|North Atlantic Ocean"]:
        for y in range(2013, 2015):
            np.testing.assert_allclose(
                res.filter(region=region, year=y).values,
                parent.filter(region=region, year=y).values,
                rtol=1e-5,
            )


def test_stitching_in_file_BCC_CSM2_MR(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_in_file_BCC_CSM2_MR"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            stitch_netcdf_scm_ncs,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "--number-workers",
                1,
                "--regexp",
                ".*BCC-CSM2-MR.*ssp126.*r1i1p1f1.*tas.*",
                "--out-format",
                "mag-files-average-year-mid-year",
            ],
        )

    assert result.exit_code == 0, result.stderr

    out_files = glob.glob(os.path.join(output_dir, "flat", "*.MAG"))
    assert len(out_files) == 1

    res = MAGICCData(out_files[0])

    _do_generic_stitched_data_tests(res)

    child_path = res.metadata["(child) netcdf-scm crunched file"]
    assert "ssp126" in child_path
    assert "r1i1p1f1" in child_path
    assert "BCC-CSM2-MR" in child_path
    assert "tas" in child_path

    child = load_scmdataframe(os.path.join(test_cmip6_crunch_output, child_path.replace("CMIP6/", "")))

    for region in ["World", "World|North Atlantic Ocean"]:
        for y in range(2015, 2017):
            np.testing.assert_allclose(
                res.filter(region=region, year=y).values,
                child.time_mean("AC").filter(region=region, year=y).values,
                rtol=1e-5,
            )

    parent_path = res.metadata["(parent) netcdf-scm crunched file"]
    assert "historical" in parent_path
    assert "r1i1p1f1" in parent_path
    assert "BCC-CSM2-MR" in parent_path
    assert "tas" in parent_path

    parent = load_scmdataframe(os.path.join(test_cmip6_crunch_output, parent_path.replace("CMIP6/", "")))

    for region in ["World", "World|North Atlantic Ocean"]:
        for y in range(2013, 2015):
            np.testing.assert_allclose(
                res.filter(region=region, year=y).values,
                parent.time_mean("AC").filter(region=region, year=y).values,
                rtol=1e-5,
            )

    warn_str = (
        "Assuming BCC metadata is wrong and branch time units are actually years, "
        "not days"
    )
    bcc_warning = [r for r in caplog.record_tuples if r[2] == warn_str]
    assert len(bcc_warning) == 1
    assert bcc_warning[0][1] == logging.WARNING


def test_stitching_no_parent(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_no_parent"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_netcdf_scm_ncs,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "--number-workers",
                1,
                "--regexp",
                ".*CNRM-ESM2-1.*r2i1p1f2.*/cSoil/.*",
            ],
        )

    assert result.exit_code != 0

    error_msg = re.compile(
        ".*No parent data \\(ssp585\\) available for "
        ".*CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-ESM2-1/ssp534-over/r2i1p1f2/Emon/cSoil/gr/v20190410/netcdf-scm_cSoil_Emon_CNRM-ESM2-1_ssp534-over_r2i1p1f2_gr_201501-210012.nc"
        ", we looked in "
        ".*CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-ESM2-1/ssp585/r2i1p1f2/Emon/cSoil/gr/\\*/netcdf-scm_cSoil_Emon_CNRM-ESM2-1_ssp585_r2i1p1f2_gr_\\*.nc"
    )
    no_parent_error = [r for r in caplog.record_tuples if error_msg.match(r[2])]
    assert len(no_parent_error) == 1
    assert no_parent_error[0][1] == logging.ERROR


def test_stitching_with_normalisation(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_with_normalisation"
    norm_method = "31-yr-mean-after-branch-time"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            stitch_netcdf_scm_ncs,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "--number-workers",
                1,
                "--regexp",
                ".*CESM2.*r10i1p1f1.*tas.*",
                "--normalise",
                norm_method
            ],
        )

    assert result.exit_code == 0

    out_files = glob.glob(os.path.join(output_dir, "flat", "*.MAG"))
    assert len(out_files) == 1

    res = MAGICCData(out_files[0])

    _do_generic_normalised_data_tests(res)
    assert res.metadata["normalisation method"] == norm_method

    child_path = res.metadata["(child) netcdf-scm crunched file"]
    assert "historical" in child_path
    assert "r10i1p1f1" in child_path
    assert "CESM2" in child_path
    assert "tas" in child_path

    child = load_scmdataframe(os.path.join(test_cmip6_crunch_output, child_path.replace("CMIP6/", "")))

    normalisation_path = res.metadata["(normalisation) netcdf-scm crunched file"]
    assert "piControl" in normalisation_path
    assert "r1i1p1f1" in normalisation_path
    assert "CESM2" in normalisation_path
    assert "tas" in normalisation_path

    normalisation = load_scmdataframe(os.path.join(test_cmip6_crunch_output, normalisation_path.replace("CMIP6/", "")))

    assert child.metadata["branch_time_in_parent"] == 306600.0
    assert child.metadata["parent_time_units"] == "days since 0001-01-01 00:00:00"
    expected_norm_year_raw = int(1 + (306600 / 365))

    norm_shift = normalisation.filter(year=range(expected_norm_year_raw, expected_norm_year_raw+31)).timeseries().mean(axis=1)

    for region in ["World", "World|North Atlantic Ocean"]:
        for y in range(1850, 2015):
            np.testing.assert_allclose(
                res.filter(region=region, year=y).values,
                child.filter(region=region, year=y).values - norm_shift[norm_shift.index.get_level_values("region") == region].values.squeeze(),
                rtol=1e-5,
            )


def test_stitching_with_normalisation_two_levels(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_with_normalisation"
    norm_method = "31-yr-mean-after-branch-time"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            stitch_netcdf_scm_ncs,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "--number-workers",
                1,
                "--regexp",
                ".*BCC-CSM2-MR.*ssp126.*tas.*",
                "--normalise",
                norm_method
            ],
        )

    assert result.exit_code == 0

    out_files = glob.glob(os.path.join(output_dir, "flat", "*.MAG"))
    assert len(out_files) == 1

    res = MAGICCData(out_files[0])

    _do_generic_stitched_data_tests(res)
    _do_generic_normalised_data_tests(res)
    assert res.metadata["normalisation method"] == norm_method

    child_path = res.metadata["(child) netcdf-scm crunched file"]
    assert "ssp126" in child_path
    assert "r1i1p1f1" in child_path
    assert "BCC-CSM2-MR" in child_path
    assert "tas" in child_path

    child = load_scmdataframe(os.path.join(test_cmip6_crunch_output, child_path.replace("CMIP6/", "")))

    parent_path = res.metadata["(parent) netcdf-scm crunched file"]
    assert "historical" in parent_path
    assert "r1i1p1f1" in child_path
    assert "BCC-CSM2-MR" in child_path
    assert "tas" in child_path

    parent = load_scmdataframe(os.path.join(test_cmip6_crunch_output, parent_path.replace("CMIP6/", "")))

    normalisation_path = res.metadata["(normalisation) netcdf-scm crunched file"]
    assert "piControl" in normalisation_path
    assert "r1i1p1f1" in normalisation_path
    assert "BCC-CSM2-MR" in normalisation_path
    assert "tas" in normalisation_path

    normalisation = load_scmdataframe(os.path.join(test_cmip6_crunch_output, normalisation_path.replace("CMIP6/", "")))

    assert parent.metadata["branch_time_in_parent"] == 2289.0
    assert parent.metadata["parent_time_units"] == "days since 1850-01-01"
    # assuming mislabelling of years as days
    expected_norm_year_raw = 2289

    norm_shift = normalisation.filter(year=range(expected_norm_year_raw, expected_norm_year_raw+31)).timeseries().mean(axis=1)

    for region in ["World", "World|North Atlantic Ocean"]:
        for y in range(2015, 2101):
            np.testing.assert_allclose(
                res.filter(region=region, year=y).values,
                child.filter(region=region, year=y).values - norm_shift[norm_shift.index.get_level_values("region") == region].values.squeeze(),
                rtol=1e-5,
            )

    for region in ["World", "World|North Atlantic Ocean"]:
        for y in range(1850, 2015):
            np.testing.assert_allclose(
                res.filter(region=region, year=y).values,
                parent.filter(region=region, year=y).values - norm_shift[norm_shift.index.get_level_values("region") == region].values.squeeze(),
                rtol=1e-5,
            )

    warn_str = (
        "Assuming BCC metadata is wrong and branch time units are actually years, "
        "not days"
    )
    bcc_warning = [r for r in caplog.record_tuples if r[2] == warn_str]
    assert len(bcc_warning) == 2
    assert bcc_warning[0][1] == logging.WARNING


def test_stitching_with_normalisation_in_file_BCC_CSM2_MR(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_with_normalisation_in_file_BCC_CSM2_MR"
    norm_method = "31-yr-mean-after-branch-time"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(
            stitch_netcdf_scm_ncs,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "--number-workers",
                1,
                "--regexp",
                ".*BCC-CSM2-MR.*1pctCO2-bgc.*r1i1p1f1.*tas.*",
                "--out-format",
                "mag-files-average-year-mid-year",
                "--normalise",
                norm_method
            ],
        )

    assert result.exit_code == 0

    out_files = glob.glob(os.path.join(output_dir, "flat", "*.MAG"))
    assert len(out_files) == 1

    res = MAGICCData(out_files[0])

    _do_generic_normalised_data_tests(res)
    assert res.metadata["normalisation method"] == norm_method

    child_path = res.metadata["(child) netcdf-scm crunched file"]
    assert "1pctCO2-bgc" in child_path
    assert "r1i1p1f1" in child_path
    assert "BCC-CSM2-MR" in child_path
    assert "tas" in child_path

    child = load_scmdataframe(os.path.join(test_cmip6_crunch_output, child_path.replace("CMIP6/", "")))

    normalisation_path = res.metadata["(normalisation) netcdf-scm crunched file"]
    assert "piControl" in normalisation_path
    assert "r1i1p1f1" in normalisation_path
    assert "BCC-CSM2-MR" in normalisation_path
    assert "tas" in normalisation_path

    normalisation = load_scmdataframe(os.path.join(test_cmip6_crunch_output, normalisation_path.replace("CMIP6/", "")))

    assert child.metadata["branch_time_in_parent"] == 0.0
    assert child.metadata["parent_time_units"] == "days since 1850-01-01"
    expected_norm_year_raw = 1850

    norm_shift = normalisation.filter(year=range(expected_norm_year_raw, expected_norm_year_raw+31)).timeseries().mean(axis=1)

    for region in ["World", "World|North Atlantic Ocean"]:
        for y in range(1850, 2015):
            np.testing.assert_allclose(
                res.filter(region=region, year=y).values,
                child.time_mean("AC").filter(region=region, year=y).values - norm_shift[norm_shift.index.get_level_values("region") == region].values.squeeze(),
                rtol=1e-5,
            )

    warn_str = (
        "Assuming BCC metadata is wrong and branch time units are actually years, "
        "not days"
    )
    bcc_warning = [r for r in caplog.record_tuples if r[2] == warn_str]
    # this file has a branch time of zero so the warning shouldn't be zero
    assert not bcc_warning


def test_stitching_with_normalisation_no_picontrol(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_with_normalisation_no_picontrol"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_netcdf_scm_ncs,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "--number-workers",
                1,
                "--regexp",
                ".*GFDL-CM4.*1pctCO2.*r1i1p1f1.*",
                "--normalise",
                "31-yr-mean-after-branch-time"
            ],
        )

    assert result.exit_code != 0
    error_msg = re.compile(
        ".*No parent data \\(piControl\\) available for "
        ".*CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/1pctCO2/r1i1p1f1/Amon/tas/gr1/v20180701/netcdf-scm_tas_Amon_GFDL-CM4_1pctCO2_r1i1p1f1_gr1_000101-015012.nc"
        ", we looked in "
        ".*CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1/Amon/tas/gr1/\\*/netcdf-scm_tas_Amon_GFDL-CM4_piControl_r1i1p1f1_gr1_\\*.nc"
    )
    no_parent_error = [r for r in caplog.record_tuples if error_msg.match(r[2])]
    assert len(no_parent_error) == 1
    assert no_parent_error[0][1] == logging.ERROR


def test_stitching_with_normalisation_no_branching_time(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_with_normalisation_no_branching_time"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_netcdf_scm_ncs,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "--number-workers",
                1,
                "--regexp",
                ".*CNRM-CM6-1.*hist-aer.*tas.*",
                "--normalise",
                "31-yr-mean-after-branch-time"
            ],
        )

    assert result.exit_code != 0
    error_msg = re.compile(
        ".*Branching time `188301` not available in piControl data in "
        "CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/piControl/r1i1p1f2/Amon/tas/gr/v20180814/netcdf-scm_tas_Amon_CNRM-CM6-1_piControl_r1i1p1f2_gr_230001-231012.nc"
    )
    no_branch_time_error = [r for r in caplog.record_tuples if error_msg.match(r[2])]
    assert len(no_branch_time_error) == 1
    assert no_branch_time_error[0][1] == logging.ERROR


def test_stitching_with_normalisation_not_enough_branching_time(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_with_normalisation_no_branching_time"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_netcdf_scm_ncs,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "--number-workers",
                1,
                "--regexp",
                ".*MIROC6.*r1i1p1f1.*rlut.*",
                "--normalise",
                "31-yr-mean-after-branch-time"
            ],
        )

    assert result.exit_code != 0
    error_msg = re.compile(
        ".*Only `320001` to `320212` is available after the branching time `320001` in piControl "
        "data in "
        "CMIP6/CMIP/MIROC/MIROC6/piControl/r1i1p1f1/Amon/rlut/gn/v20181212/netcdf-scm_rlut_Amon_MIROC6_piControl_r1i1p1f1_gn_320001-320212.nc"
    )
    not_enough_norm_data_error = [r for r in caplog.record_tuples if error_msg.match(r[2])]
    assert len(not_enough_norm_data_error) == 1
    assert not_enough_norm_data_error[0][1] == logging.ERROR


@pytest.mark.parametrize(
    "out_format",
    (
        "magicc-input-files",
        "magicc-input-files-average-year-start-year",
        "magicc-input-files-average-year-mid-year",
        "magicc-input-files-average-year-end-year",
        "magicc-input-files-point-start-year",
        "magicc-input-files-point-mid-year",
        "magicc-input-files-point-end-year",
        "mag-files",
        "mag-files-average-year-start-year",
        "mag-files-average-year-mid-year",
        "mag-files-average-year-end-year",
        "mag-files-point-start-year",
        "mag-files-point-mid-year",
        "mag-files-point-end-year",
    ),
)
def test_stitching_file_types(tmpdir, caplog, test_cmip6_crunch_output, out_format):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_netcdf_scm_ncs,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "--out-format",
                out_format,
                "-f",
                "--number-workers",
                1,
                "--regexp",
                ".*BCC-CSM2-MR.*(ssp126|historical).*tas.*",
                "--normalise",
                "31-yr-mean-after-branch-time"
            ],
        )

    assert result.exit_code == 0

    out_files = glob.glob(os.path.join(output_dir, "flat", "*.IN" if out_format.startswith("magicc") else "*.MAG"))
    assert len(out_files) == 2

    for p in out_files:
        res = MAGICCData(p)
        _do_generic_normalised_data_tests(res)
        if "ssp26" in p:
            _do_generic_stitched_data_tests(res)


def test_prefix():
    assert False


def test_target_units():
    assert False

