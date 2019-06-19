#!/bin/bash
SRC_DIR="/media/research-nfs/cmip6/"
CRUNCH_DIR="/home/UNIMELB/znicholls/test-cmip6-crunch-short"
WRANGLE_DIR="/home/UNIMELB/znicholls/test-cmip6-wrangle-short"
CONTACT='zebedee nicholls <zebedee.nicholls@climate-energy-college.org>'
DRS="CMIP6Output"
REGEXP="^(?!.*fx).*IPSL.*$"
# have to be super careful when crunching input files as the duplicate grids can cause
# things to explode
REGEXP_WRANGLE_IN_FILES=".*tas/.*gr/.*"

netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}"
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}"
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" --out-format magicc-input-files-point-end-of-year --regexp "${REGEXP_WRANGLE_IN_FILES}"
