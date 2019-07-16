#!/bin/bash
SRC_DIR="tests/test-data/cmip6output"
CRUNCH_DIR="output-examples/scratch-process-output"
WRANGLE_DIR="output-examples/wrangle-process-output"
CONTACT='Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>'
DRS="CMIP6Output"
REGEXP="^(?!.*fx).*IPSL.*$"
# have to be super careful when crunching input files as the duplicate grids can cause
# things to explode
REGEXP_WRANGLE_IN_FILES=".*tas/.*gr/.*"

netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" -f
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" -f
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" --out-format magicc-input-files-point-end-of-year --regexp "${REGEXP_WRANGLE_IN_FILES}" -f
