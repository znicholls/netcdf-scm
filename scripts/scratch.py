import sys
from datetime import datetime

from click.testing import CliRunner

from netcdf_scm.cli import crunch_data

INPUT_DIR = "/data/marble/cmip6/CMIP6/DCPP/IPSL/IPSL-CM6A-LR/dcppC-atl-control/r1i1p1f1/Amon/tas/gr/v20190110"
#INPUT_DIR = "/data/marble/cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r1i1p1f1/Amon/tas/gn/v20181126"
#INPUT_DIR = "/data/marble/cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/piControl/r1i1p1f1/Amon/tas/gn/v20181016"
INPUT_DIR = sys.argv[1]
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
