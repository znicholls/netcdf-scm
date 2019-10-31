#!/bin/bash
SRC_DIR="/data/marble/cmip6/CMIP6/"
# CRUNCH_DIR="/data/marble/sandbox/share/cmip6-crunched-full-test-with-units"
CRUNCH_DIR="/data/marble/sandbox/share/cmip6-crunched-ipsl-units-test"
# WRANGLE_DIR="/data/marble/sandbox/share/cmip6-wrangled-full-test-with-units"
WRANGLE_DIR="/data/marble/sandbox/share/cmip6-wrangled-ipsl-units-test"
# WRANGLE_DIR_ANNUAL_MEAN="/data/marble/sandbox/share/cmip6-wrangled-full-test-with-units/annual-mean-mag-files"
WRANGLE_DIR_ANNUAL_MEAN="/data/marble/sandbox/share/cmip6-wrangled-ipsl-units-test/annual-mean-mag-files"
WRANGLE_UNITS_SPECS_CSV="./units-specs.csv"

CONTACT='Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>'
DRS="CMIP6Output"

REGIONS="World,World|Northern Hemisphere,World|Southern Hemisphere,World|Land,World|Ocean,World|Northern Hemisphere|Land,World|Southern Hemisphere|Land,World|Northern Hemisphere|Ocean,World|Southern Hemisphere|Ocean,World|El Nino N3.4,World|North Atlantic Ocean"

# REGEXP="^(?!.*(fx|/ta/|/co2/|tos)).*(/ta/|/co2/|/fco2fos/|/fco2nat/|/fco2antt/|/co2s/|/nMineral/|/fBNF/|/fNnetmin/|/fNdep/|/fNfert/|/fNloss/|/fNup/|/fNgas/|/nep/|/netAtmosLandCO2Flux/|/nppGrass/|/nppLut/|/nppOther/|/nppShrub/|/nppStem/|/nppTree/|/rhGrass/|/rhLitter/|/rhLut/|/rhShrub/|/rhSoil/|/rhTree/|/cLand/|/cSoil/|/cVegGrass/|/cVegShrub/|/cVegTree/|/cLitterSurf/|/cLitterSubSurf/|/cLitterCWD/|/cLitterGrass/|/cLitterShrub/|/cLitterTree/|/fAnthDisturb/|/fLuc/|/fLuccAtmoLut/|/fNLandToOcean/|/landCoverFrac/|/cLitter/|/cVeg/|/cSoilFast/|/cSoilMedium/|/cSoilSlow/|/gpp/|/nbp/|/npp/|/lai/|/ra/|/rh/|/fFire/|/fgco2/|/thetao/|/hfds/|/hfcorr/|/tos/|/zos/|/so/|/cLitter/|/cProduct/|/cSoil/|/cSoilFast/|/cSoilMedium/|/cSoilSlow/|/cVeg/|/fFire/|/fGrazing/|/fHarvest/|/fLitterSoil/|/fLuc/|/fVegLitter/|/fVegSoil/|/pr/|/prsn/|/rlut/|/rsdt/|/rsut/|/tas/|/tasmin/|/tasmax/|/ts/).*$"
REGEXP="^(?!.*(fx|/ta/|/co2/)).*IPSL.*r1i1p1f1.*(/cLand/|/cLitter/|/cLitterGrass/|/cLitterSubSurf/|/cLitterSurf/|/cLitterTree/|/cProduct/|/cSoil/|/cSoilFast/|/cSoilMedium/|/cSoilSlow/|/cVeg/|/cVegGrass/|/cVegShrub/|/cVegTree/|/co2s/|/fAnthDisturb/|/fBNF/|/fFire/|/fGrazing/|/fHarvest/|/fLitterSoil/|/fLuc/|/fNdep/|/fNfert/|/fNgas/|/fNloss/|/fNnetmin/|/fNup/|/fVegLitter/|/fco2antt/|/fco2fos/|/fco2nat/|/fgco2/|/gpp/|/hfds/|/lai/|/nMineral/|/nbp/|/nep/|/netAtmosLandCO2Flux/|/npp/|/nppGrass/|/nppOther/|/nppShrub/|/nppStem/|/nppTree/|/pr/|/prsn/|/ra/|/rh/|/rhGrass/|/rhLitter/|/rhSoil/|/rhTree/|/rlut/|/rsdt/|/rsut/|/tas/|/tasmax/|/tasmin/|/tos/|/ts/|/zos/).*$"

# have to be super careful when crunching input files as the duplicate grids can cause
# things to explode
REGEXP_WRANGLE_IN_FILES=".*tas/.*gr/.*"

#netcdf-scm-crunch "${SRC_DIR}" "${CRUNCH_DIR}" "${CONTACT}"  --drs "${DRS}" --regexp "${REGEXP}" --regions "${REGIONS}" --small-number-workers 10 --medium-number-workers 3

# netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" --number-workers 1 --target-units-specs "${WRANGLE_UNITS_SPECS_CSV}"

netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR_ANNUAL_MEAN}" "${CONTACT}" --drs "${DRS}" --out-format mag-files-average-beginning-of-year --number-workers 1 --target-units-specs "${WRANGLE_UNITS_SPECS_CSV}"

# netcdf-scm-wrangle "${CRUNCH_DIR}/netcdf-scm-crunched/CMIP6/" "${WRANGLE_DIR}" "${CONTACT}" --drs "${DRS}" --out-format magicc-input-files-point-end-of-year --regexp "${REGEXP_WRANGLE_IN_FILES}" --number-workers 1
