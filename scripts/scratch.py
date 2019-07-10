from os.path import join

from click.testing import CliRunner

from netcdf_scm.cli import crunch_data


INPUT_DIR = "tests/test-data/cmip6output/CMIP6/"
OUTPUT_DIR = "output-examples/scratch-script-output"

runner = CliRunner()
result = runner.invoke(
    crunch_data,
    [
        INPUT_DIR,
        OUTPUT_DIR,
        "cmip6output crunching scratch",
        "--drs",
        "CMIP6Output",
        "-f",
    ],
)
assert result.exit_code == 0, result.output
