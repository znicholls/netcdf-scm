import glob
import os.path

import pytest
from click.testing import CliRunner
from pymagicc.io import MAGICCData

from netcdf_scm.cli import stitch_data

# TODO:
# - create small (i.e. regridded using cdo remapbil,n4 <infile> <outfile>) test files which don't clash with existing data and match all the tests
    # - test data (timeseries which can be stitched, one ssp plus hist, one hist plus piControl, one 1pctCO2 plus piControl, one BCC-CSM2-MR hist plus BCC-CSM2-MR piControl)
# - start passing tests
    # - tests of errors when parent isn't there
    # - tests of errors when branch time isn't in data
    # - tests of errors when not enough years in piControl are there
    # - tests of log messages when doing the BCC-CSM2-MR fix
    # - tests of normalisation or not


def test_stitching_default(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_default"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_data,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "---number-workers",
                1,
                "--regexp",
                ".*EC-Earth3-Veg.*ssp585.*r1i1p1f1.*hfds.*",
            ],
        )

    assert result.exit_code == 0

    out_files = glob.glob(os.path.join(output_dir, "flat", "*.MAG"))
    assert len(out_files) == 1

    res = MAGICCData(out_files[0])

    assert False, "do data tests here, join of SSP585 and hist"
    assert False, "do metadata tests here, parent"
    assert False, "do tests of logs here, shouldn't be any problems"


def test_stitching_in_file_BCC_CSM2_MR(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_default"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_data,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "---number-workers",
                1,
                "--regexp",
                ".*BCC-CSM2-MR.*ssp126.*r1i1p1f1.*tas.*",
                "--out-format",
                "mag-files-average-year-mid-year",
            ],
        )

    assert result.exit_code == 0

    out_files = glob.glob(os.path.join(output_dir, "flat", "*.MAG"))
    assert len(out_files) == 1

    res = MAGICCData(out_files[0])

    assert False, "do data tests here, join of SSP and hist"
    assert False, "do metadata tests here, parent"
    assert False, "do tests of logs here, should log that we're assuming branch time means year rather than days since"


def test_stitching_no_parent(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_no_parent"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_data,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "---number-workers",
                1,
                "--regexp",
                ".*CNRM-ESM2-1.*r2i1p1f2.*/cSoil/.*",
            ],
        )

    assert result.exit_code != 0
    assert str(result.exception) == "No parent data available for filename"


def test_stitching_with_normalisation(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_default"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_data,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "---number-workers",
                1,
                "--regexp",
                ".*CESM2.*r10i1p1f1.*tas.*",
                "--normalise",
                "31-yr-mean-after-branch-time"
            ],
        )

    assert result.exit_code == 0

    out_files = glob.glob(os.path.join(output_dir, "flat", "*.MAG"))
    assert len(out_files) == 1

    res = MAGICCData(out_files[0])

    assert False, "do data tests here, normalised hist"
    assert False, "do metadata tests here, parent and normalisation experiment"
    assert False, "do tests of logs here, shouldn't be any problems"


def test_stitching_with_normalisation_in_file_BCC_CSM2_MR(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_default"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_data,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "---number-workers",
                1,
                "--regexp",
                ".*BCC-CSM2-MR.*1pctCO2-bgc.*tas.*r1i1p1f1.*",
                "--out-format",
                "mag-files-average-year-mid-year",
                "--normalise",
                "31-yr-mean-after-branch-time"
            ],
        )

    assert result.exit_code == 0

    out_files = glob.glob(os.path.join(output_dir, "flat", "*.MAG"))
    assert len(out_files) == 1

    res = MAGICCData(out_files[0])

    assert False, "do data tests here, normalised hist"
    assert False, "do metadata tests here, parent and normalisation here"
    assert False, "do tests of logs here, should log that we're assuming branch time means year rather than days since"


def test_stitching_with_normalisation_no_picontrol(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_with_normalisation_no_picontrol"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_data,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "---number-workers",
                1,
                "--regexp",
                ".*GFDL-CM4.*1pctCO2.*r1i1p1f1.*",
                "--normalise",
                "31-yr-mean-after-branch-time"
            ],
        )

    assert result.exit_code != 0
    assert str(result.exception) == "No parent data available for filename"


def test_stitching_with_normalisation_no_branching_time(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_with_normalisation_no_branching_time"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_data,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "---number-workers",
                1,
                "--regexp",
                ".*UKESM1-0-LL.*historical.*hfds.*",
                "--normalise",
                "31-yr-mean-after-branch-time"
            ],
        )

    assert result.exit_code != 0
    assert str(result.exception) == "Branching time `{}` not available in piControl data".format(2014)


def test_stitching_with_normalisation_not_enough_branching_time(tmpdir, caplog, test_cmip6_crunch_output):
    output_dir = str(tmpdir)
    crunch_contact = "test_stitching_with_normalisation_no_branching_time"

    runner = CliRunner(mix_stderr=False)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            stitch_data,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "-f",
                "---number-workers",
                1,
                "--regexp",
                ".*MIROC6.*rlut.*r1i1p1f1.*",
                "--normalise",
                "31-yr-mean-after-branch-time"
            ],
        )

    assert result.exit_code != 0
    assert str(result.exception) == "Only {} years of data are available after the branching time (`{}`) in the piControl data".format(3, 2014)


@pytest.mark.parametrize(
    "out_format",
    (
        "magicc-input-files-average-year-start-year",
        "magicc-input-files-average-year-mid-year",
        "magicc-input-files-average-year-end-year",
        "magicc-input-files-point-start-year",
        "magicc-input-files-point-mid-year",
        "magicc-input-files-point-end-year",
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
            stitch_data,
            [
                test_cmip6_crunch_output,
                output_dir,
                crunch_contact,
                "--drs",
                "CMIP6Output",
                "--out-format",
                out_format,
                "-f",
                "---number-workers",
                1,
                # will need some regexp here to make things work
            ],
        )

    assert result.exit_code == 0

    assert len(os.path.join(output_dir, "flat", "*.IN" if out_format.startswith("magicc") else ".MAG")) == 5
    assert False, "do metadata tests here, parent"
