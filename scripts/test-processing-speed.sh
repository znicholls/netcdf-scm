#!/bin/bash
SRC_DIR="/data/marble/cmip6/CMIP6/DCPP/IPSL/IPSL-CM6A-LR/dcppC-ipv-neg"
CRUNCH_DIR="/data/marble/sandbox/share/cmip6-crunched-ipsl-sandbox"
CONTACT='Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>'
DRS="CMIP6Output"
REGEXP="^(?!.*fx).*(tas/|fco2nat|rsut|rlut|rsdt).*$"

exec 3>&1 4>&2
SRC_DIR_TIME=$( { time netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --force 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-


SRC_DIR_MEDIUM="/data/marble/cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r1i1p1f1/Amon"

exec 3>&1 4>&2
SRC_DIR_MEDIUM_TIME=$( { time netcdf-scm-crunch "${SRC_DIR_MEDIUM}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --force 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-


SRC_DIR_BIG="/data/marble/cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/piControl/r1i1p1f1/Amon"

exec 3>&1 4>&2
SRC_DIR_BIG_TIME=$( { time netcdf-scm-crunch "${SRC_DIR_BIG}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --force 1>&3 2>&4; } 2>&1 )
exec 3>&- 4>&-

echo "Multiple small file (10 yrs) time: $SRC_DIR_TIME"
echo "Multiple medium file (250 yrs) crunch time: $SRC_DIR_MEDIUM_TIME"
echo "Multiple big file (750 yrs) crunch time: $SRC_DIR_BIG_TIME"
