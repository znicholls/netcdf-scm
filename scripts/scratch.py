from click.testing import CliRunner

from netcdf_scm.cli import crunch_data

runner = CliRunner()
result = runner.invoke(
    crunch_data,
    [
        "tests/test-data/marble-cmip5/cmip5/rcp26/Amon/tas/bcc-csm1-1/r1i1p1",
        "/tmp",
        "--cube-type",
        "MarbleCMIP5",
        "-f",
    ],
)

print(result.output)
