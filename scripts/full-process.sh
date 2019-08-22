#!/bin/bash
SRC_DIR="/data/marble/cmip6/CMIP6"
CRUNCH_DIR="/data/marble/sandbox/share/cmip6-crunched-full-test"
WRANGLE_DIR="/data/marble/sandbox/share/cmip6-wrangled-full-test"
CONTACT='Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>'
DRS="CMIP6Output"
# REGEXP="^(?!.*(fx|/ta/|/co2/|tos)).*(tas/|fco2nat|rsut|rlut|rsdt|ts|fgco2|hfds|hfcorr|tos|fco2fos|fco2nat|fco2antt|co2fs|hfds|thetao).*gr.*$"
REGEXP="^(?!.*(fx|/ta/|/co2/|tos)).*(/ta/|/co2/|/fco2fos/|/fco2nat/|/fco2antt/|/co2s/|/nMineral/|/fBNF/|/fNnetmin/|/fNdep/|/fNfert/|/fNloss/|/fNup/|/fNgas/|/nep/|/netAtmosLandCO2Flux/|/nppGrass/|/nppLut/|/nppOther/|/nppShrub/|/nppStem/|/nppTree/|/rhGrass/|/rhLitter/|/rhLut/|/rhShrub/|/rhSoil/|/rhTree/|/cLand/|/cSoil/|/cVegGrass/|/cVegShrub/|/cVegTree/|/cLitterSurf/|/cLitterSubSurf/|/cLitterCWD/|/cLitterGrass/|/cLitterShrub/|/cLitterTree/|/fAnthDisturb/|/fLuc/|/fLuccAtmoLut/|/fNLandToOcean/|/landCoverFrac/|/cLitter/|/cVeg/|/cSoilFast/|/cSoilMedium/|/cSoilSlow/|/gpp/|/nbp/|/npp/|/lai/|/ra/|/rh/|/fFire/|/fgco2/|/thetao/|/hfds/|/hfcorr/|/tos/|/zos/|/so/|/cLitter/|/cProduct/|/cSoil/|/cSoilFast/|/cSoilMedium/|/cSoilSlow/|/cVeg/|/fFire/|/fGrazing/|/fHarvest/|/fLitterSoil/|/fLuc/|/fVegLitter/|/fVegSoil/|/pr/|/prsn/|/rlut/|/rsdt/|/rsut/|/tas/|/tasmin/|/tasmax/|/ts/).*$"

# have to be super careful when crunching input files as the duplicate grids can cause
# things to explode
REGEXP_WRANGLE_IN_FILES=".*tas/.*gr/.*"

netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --small-number-workers 10 --medium-number-workers 3
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" --number-workers 1
netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" --out-format magicc-input-files-point-end-of-year --regexp "${REGEXP_WRANGLE_IN_FILES}" --number-workers 1
