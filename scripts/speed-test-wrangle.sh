#!/bin/bash
CRUNCH_DIR="output-examples/wrangling-speed-test"
CONTACT='Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>'
DRS="CMIP6Output"


SRC_DIR="./tests/test-data/expected-crunching-output/cmip6output/CMIP6/"
NUMBER_WORKERS=15

exec 3>&1 4>&2
SRC_DIR_TIME_SERIAL=$( { time netcdf-scm-wrangle "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --force --number-workers 1 --out-format "mag-files" 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-
echo "Multiple files serially: $SRC_DIR_TIME_SERIAL"

exec 3>&1 4>&2
SRC_DIR_TIME=$( { time netcdf-scm-wrangle "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --force --number-workers ${NUMBER_WORKERS} --out-format "mag-files" 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-
echo "Multiple files: $SRC_DIR_TIME"

exec 3>&1 4>&2
SRC_DIR_TIME_AVERAGE=$( { time netcdf-scm-wrangle "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --force --number-workers ${NUMBER_WORKERS} --out-format "mag-files-average-year-mid-year" 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-
echo "Multiple files with average time: $SRC_DIR_TIME_AVERAGE"

echo ""
echo ""

exec 3>&1 4>&2
VERSION=$( { python -c "import netcdf_scm; import click; click.echo('NetCDF-SCM version: {}'.format(netcdf_scm.__version__))" 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-

echo "$VERSION"
echo "Wrangling multiple files wrangled serially time: $SRC_DIR_TIME_SERIAL"
echo "Wrangling multiple files time: $SRC_DIR_TIME"
echo "Wrangling multiple files with average time: $SRC_DIR_TIME_AVERAGE"
