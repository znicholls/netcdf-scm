#!/bin/bash

SRC_DIR="/media/research-nfs/cmip6/CMIP6/CMIP/"
CRUNCH_DIR="/home/UNIMELB/znicholls/test-cmip6-crunch-short"
WRANGLE_DIR="/home/UNIMELB/znicholls/test-cmip6-wrangle-short"
CONTACT='zebedee nicholls <zebedee.nicholls@climate-energy-college.org>'
DRS="CMIP6Output"
REGEXP="^(?!.*fx).*IPSL.*1pctCO2.*$"

netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}"
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}"
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" --out-format magicc-input-files-point-end-of-year --regexp ".*tas/.*gr/.*"
