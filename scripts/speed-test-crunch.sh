#!/bin/bash
CRUNCH_DIR="/home/ubuntu/data/cmip6-speed-test-output"
CONTACT='CMIP6 data crunching speed test'
DRS="CMIP6Output"
REGEXP='^(?!.*(fx|/co2/|/ta/)).*Amon.*$'

SRC_DIR_ROOT="/home/ubuntu/data/cmip6-speed-test"

SRC_DIR="${SRC_DIR_ROOT}/CMIP6/DCPP/IPSL/IPSL-CM6A-LR/dcppC-atl-control"

exec 3>&1 4>&2
SRC_DIR_TIME=$( { time netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --force --small-number-workers 15 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-
echo "Multiple small file (2.4 million data points) time: $SRC_DIR_TIME"


SRC_DIR_MEDIUM="${SRC_DIR_ROOT}/CMIP6/CMIP/BCC/BCC-CSM2-MR/historical"

exec 3>&1 4>&2
SRC_DIR_MEDIUM_TIME=$( { time netcdf-scm-crunch "${SRC_DIR_MEDIUM}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --force --small-number-workers 10 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-
echo "Multiple medium file (100 million data points) crunch time: $SRC_DIR_MEDIUM_TIME"


SRC_DIR_BIG="${SRC_DIR_ROOT}/CMIP6/CMIP/BCC/BCC-CSM2-MR/piControl/r1i1p1f1/Amon"

exec 3>&1 4>&2
SRC_DIR_BIG_TIME=$( { time netcdf-scm-crunch "${SRC_DIR_BIG}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --force --medium-number-workers 2 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-
echo "Multiple large files (300 million data points) crunch time: $SRC_DIR_BIG_TIME"

# Ocean data, future problem

# SRC_DIR_OCEAN_2D="${SRC_DIR_ROOT}//CMIP6"

# exec 3>&1 4>&2
# SRC_DIR_OCEAN_2D_TIME=$( { time netcdf-scm-crunch "${SRC_DIR_OCEAN_2D}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp '.*zos.*' --force --medium-number-workers 2 1>&3 2>&4; } 2>&1 )
# exec 3>&- 4>&-
# echo "Multiple ocean 2D files (240 million data points) crunch time: $SRC_DIR_OCEAN_2D_TIME"


# SRC_DIR_ENORMOUS="${SRC_DIR_ROOT}//CMIP6"

# exec 3>&1 4>&2
# SRC_DIR_ENORMOUS_TIME=$( { time netcdf-scm-crunch "${SRC_DIR_ENORMOUS}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp '.*thetao.*' --force --medium-number-workers 2 1>&3 2>&4; } 2>&1 )
# exec 3>&- 4>&-
# echo "Multiple enormous files (14 000 million data points) crunch time: $SRC_DIR_ENORMOUS_TIME"

echo ""
echo ""

exec 3>&1 4>&2
VERSION=$( { python -c "import netcdf_scm; import click; click.echo('NetCDF-SCM version: {}'.format(netcdf_scm.__version__))" 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-

echo "$VERSION"
echo "Multiple small file (2.4 million data points) time: $SRC_DIR_TIME"
echo "Multiple medium file (100 million data points) crunch time: $SRC_DIR_MEDIUM_TIME"
echo "Multiple big file (300 million data points) crunch time: $SRC_DIR_BIG_TIME"
# echo "Multiple ocean 2D file (240 million data points) crunch time: $SRC_DIR_OCEAN_2D_TIME"
# echo "Multiple enormous file (14 000 million data points) crunch time: $SRC_DIR_ENORMOUS_TIME"
