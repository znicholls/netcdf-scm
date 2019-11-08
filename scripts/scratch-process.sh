#!/bin/bash
SRC_DIR="tests/test-data/cmip6output/"
CRUNCH_DIR="output-examples/scratch-process-output"
WRANGLE_DIR="output-examples/wrangle-process-output"
STITCH_DIR="output-examples/stitch-process-output"
CONTACT='Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>'
DRS="CMIP6Output"

REGEXP="^(?!.*(fx|thetao)).*$"
# can't handle thetao wrangling/stitching yet...
REGEXP_WRANGLE_IN_FILES=".*BCC-CSM2-MR.*tas"
REGEXP_STITCH_IN_FILES=".*BCC-CSM2-MR.*tas"

netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" -f
netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" -f --small-number-workers 1

netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" -f
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" --out-format magicc-input-files-point-end-year --regexp "${REGEXP_WRANGLE_IN_FILES}" -f

netcdf-scm-stitch "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${STITCH_DIR}" "${CONTACT}" --drs "${DRS}" --regexp "${REGEXP_STITCH_IN_FILES}" -f
netcdf-scm-stitch "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${STITCH_DIR}" "${CONTACT}" --drs "${DRS}" --out-format magicc-input-files-point-end-year --regexp "${REGEXP_STITCH_IN_FILES}" -f
