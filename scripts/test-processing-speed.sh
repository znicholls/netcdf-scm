#!/bin/bash
CRUNCH_DIR="/data/marble/sandbox/share/cmip6-crunched-ipsl-sandbox-speed-test"
CONTACT='Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>'
DRS="CMIP6Output"
REGEXP="^(?!.*(fx|/ta/|/co2/)).*Amon.*$"


SRC_DIR="/data/marble/cmip6/CMIP6/DCPP/IPSL/IPSL-CM6A-LR/dcppC-atl-control"

exec 3>&1 4>&2
SRC_DIR_TIME=$( { time netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --force --small-number-workers 15 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-
echo "Multiple small file (2.4 million data points) time: $SRC_DIR_TIME"


# SRC_DIR_MEDIUM="/data/marble/cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/historical"

# exec 3>&1 4>&2
# SRC_DIR_MEDIUM_TIME=$( { time netcdf-scm-crunch "${SRC_DIR_MEDIUM}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --force --small-number-workers 10 1>&3 2>&4; } 2>&1 )
# exec 3>&- 4>&-
# echo "Multiple medium file (100 million data points) crunch time: $SRC_DIR_MEDIUM_TIME"


# SRC_DIR_BIG="/data/marble/cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/piControl/r1i1p1f1/Amon"

# exec 3>&1 4>&2
# SRC_DIR_BIG_TIME=$( { time netcdf-scm-crunch "${SRC_DIR_BIG}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --force --medium-number-workers 2 1>&3 2>&4; } 2>&1 )
# exec 3>&- 4>&-
# echo "Multiple large files (300 million data points) crunch time: $SRC_DIR_BIG_TIME"


SRC_DIR_OCEAN_2D="/data/marble/sandbox/znicholls/test-cmip6output/CMIP6"

exec 3>&1 4>&2
SRC_DIR_OCEAN_2D_TIME=$( { time netcdf-scm-crunch "${SRC_DIR_OCEAN_2D}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp ".*zos.*" --force --medium-number-workers 2 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-
echo "Multiple ocean 2D files (240 million data points) crunch time: $SRC_DIR_OCEAN_2D"


SRC_DIR_ENORMOUS="/data/marble/sandbox/znicholls/test-cmip6output/CMIP6"

exec 3>&1 4>&2
SRC_DIR_ENORMOUS_TIME=$( { time netcdf-scm-crunch "${SRC_DIR_ENORMOUS}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp ".*thetao.*" --force --medium-number-workers 2 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-
echo "Multiple enormous files (14 000 million data points) crunch time: $SRC_DIR_ENORMOUS_TIME"


exec 3>&1 4>&2
VERSION=$( { python -c "import netcdf_scm; import click; click.echo('NetCDF-SCM version: {}'.format(netcdf_scm.__version__))" 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-

echo "NetCDF-SCM version: $VERSION"
echo "Multiple small file (2.4 million data points) time: $SRC_DIR_TIME"
echo "Multiple medium file (100 million data points) crunch time: $SRC_DIR_MEDIUM_TIME"
echo "Multiple big file (300 million data points) crunch time: $SRC_DIR_BIG_TIME"
echo "Multiple ocean 2D file (240 million data points) crunch time: $SRC_DIR_OCEAN_2D_TIME"
echo "Multiple enormous file (14 000 million data points) crunch time: $SRC_DIR_ENORMOUS_TIME"
