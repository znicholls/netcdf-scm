#!/bin/bash
SRC_DIR="/media/research-nfs/cmip6"
CRUNCH_DIR="/data/marble/sandbox/share/cmip6-crunched-ipsl-sandbox"
WRANGLE_DIR="/data/marble/sandbox/share/cmip6-wrangled-ipsl-sandbox"
CONTACT='Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>'
DRS="CMIP6Output"
REGEXP="^(?!.*(fx|/ta/)).*(rlut|rsdt|rsut|tas|ts|fgco2|hfds|hfcorr|tos|co2|fco2fos|fco2nat|fco2antt|co2s).*$"
# have to be super careful when crunching input files as the duplicate grids can cause
# things to explode
REGEXP_WRANGLE_IN_FILES=".*tas/.*gr/.*"

netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" 
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}"
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" --out-format magicc-input-files-point-end-of-year --regexp "${REGEXP_WRANGLE_IN_FILES}"
