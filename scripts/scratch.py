import sys
from datetime import datetime

from click.testing import CliRunner

from netcdf_scm.cli import crunch_data

try:
    INPUT_DIR = sys.argv[1]
except IndexError:
    INPUT_DIR = "tests/test-data/cmip6output"
OUTPUT_DIR = "output-examples/scratch-script-output"

start = datetime.now()
runner = CliRunner()
for _ in range(5):
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
total_time = (end - start).total_seconds()
print(f"Start: {start}\nEnd: {end}\nDiff: {total_time}")
