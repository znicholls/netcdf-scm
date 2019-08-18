#!/bin/bash
TEST_DIR="./tests/test-data/cmip6output"
ROOT_DIR="/data/marble/cmip6/CMIP6/CMIP/NCAR/CESM2/historical/r7i1p1f1"
ROOT_DIR_TO_MAKE="CMIP6/CMIP/NCAR/CESM2/historical/r7i1p1f1"

declare -a arr=(
  "Omon/hfds/gr/v20190311/hfds_Omon_CESM2_historical_r7i1p1f1_gr_195001-199912.nc"
  "Omon/hfds/gn/v20190311/hfds_Omon_CESM2_historical_r7i1p1f1_gn_195001-199912.nc"
  "Ofx/areacello/gr/v20190311/areacello_Ofx_CESM2_historical_r7i1p1f1_gr.nc"
  "Ofx/areacello/gn/v20190311/areacello_Ofx_CESM2_historical_r7i1p1f1_gn.nc"
  "Ofx/sftof/gn/v20190311/sftof_Ofx_CESM2_historical_r7i1p1f1_gn.nc"
)

## now loop through the above array
echo "mkdir -p ${TEST_DIR}/${ROOT_DIR_TO_MAKE}"
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
   echo "mkdir -p ${TEST_DIR}/${ROOT_DIR_TO_MAKE}/${dir_to_make}"
   echo "mv $outfile ${TEST_DIR}/${ROOT_DIR_TO_MAKE}/${dir_to_make}/${outfile}"
done
