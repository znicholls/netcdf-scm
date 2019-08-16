#!/bin/bash
ROOT_DIR="/data/marble/cmip6/CMIP6/CMIP/NCAR/CESM2/historical/r7i1p1f1/"

declare -a arr=(
  "Lmon/gpp/gn/v20190311/gpp_Lmon_CESM2_historical_r7i1p1f1_gn_195001-199912.nc"
  "Amon/tas/gn/v20190311/tas_Amon_CESM2_historical_r7i1p1f1_gn_195001-199912.nc"
  "Omon/hfds/gn/v20190311/hfds_Omon_CESM2_historical_r7i1p1f1_gn_195001-199912.nc"
  "Omon/thetao/gn/v20190311/thetao_Omon_CESM2_historical_r7i1p1f1_gn_195001-199912.nc"
  "fx/areacella/gn/v20190311/areacella_fx_CESM2_historical_r7i1p1f1_gn.nc"
  "Ofx/areacello/gn/v20190311/areacello_Ofx_CESM2_historical_r7i1p1f1_gn.nc"
  "Ofx/volcello/gn/v20190311/volcello_Ofx_CESM2_historical_r7i1p1f1_gn.nc"
  "fx/sftlf/gn/v20190311/sftlf_fx_CESM2_historical_r7i1p1f1_gn.nc"
  "Ofx/sftof/gn/v20190311/sftof_Ofx_CESM2_historical_r7i1p1f1_gn.nc"
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
     cp $file ${outfile}
   else
      tmpfile=tmp.nc
      outfile=${basename/195001-199912/195701-195703}
      cdo -selyear,1957 $file $tmpfile > /dev/null 2>&1
      cdo -selmonth,1/3 $tmpfile $outfile > /dev/null 2>&1
      rm $tmpfile > /dev/null 2>&1
   fi
   echo "mkdir -p ${dir_to_make}"
   echo "mv $outfile ${dir_to_make}/${outfile}"
done
