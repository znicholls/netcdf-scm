#!/bin/bash
ROOT_DIR="/data/marble/cmip6/CMIP6/CMIP/CNRM-CERFACS/CNRM-ESM2-1/historical/r2i1p1f2/"
START_YEAR=1997
END_YEAR=1999
declare -a arr=(
    "Amon/tas/gr/v20190125/tas_Amon_CNRM-ESM2-1_historical_r2i1p1f2_gr_185001-201412.nc"
    "Lmon/cSoilFast/gr/v20190125/cSoilFast_Lmon_CNRM-ESM2-1_historical_r2i1p1f2_gr_185001-201412.nc"
    "Lmon/gpp/gr/v20190125/gpp_Lmon_CNRM-ESM2-1_historical_r2i1p1f2_gr_185001-201412.nc"
    "fx/areacella/gr/v20190125/areacella_fx_CNRM-ESM2-1_historical_r2i1p1f2_gr.nc"
    "Omon/hfds/gn/v20190125/hfds_Omon_CNRM-ESM2-1_historical_r2i1p1f2_gn_185001-201412.nc"
    "Ofx/areacello/gn/v20190125/areacello_Ofx_CNRM-ESM2-1_historical_r2i1p1f2_gn.nc"
)

## now loop through the above array
echo "mkdir -p ${ROOT_DIR}"
for i in "${arr[@]}"
do
   file="${ROOT_DIR}/$i"
   dir_to_make=$(dirname "${i}")
   basename=${i##*/}
   if [[ $file == *"fx"* ]]; then
     outfile=$basename
     cp $file ~/${outfile}
   else
      outfile=${basename/185001-201412/${START_YEAR}01-${END_YEAR}12}
      cdo selyear,${START_YEAR}/${END_YEAR} $file ~/$outfile > /dev/null 2>&1
   fi
   echo "mkdir -p ${dir_to_make}"
   echo "cp -r ~/$outfile ${dir_to_make}/${outfile}" 
done
