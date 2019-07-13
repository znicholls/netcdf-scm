from datetime import datetime

from click.testing import CliRunner

from netcdf_scm.cli import crunch_data

INPUT_DIR = "tests/test-data/cmip6output/CMIP6/"
OUTPUT_DIR = "output-examples/scratch-script-output"

start = datetime.now()
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
end = datetime.now()
print(result.output)
total_time = (end-start).total_seconds()
print(f"Start: {start}\nEnd: {end}\nDiff: {total_time}")
