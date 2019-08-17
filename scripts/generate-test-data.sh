#!/bin/bash
TEST_DIR="./tests/test-data/cmip6output"
ROOT_DIR="/data/marble/cmip6/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1"
ROOT_DIR_TO_MAKE="CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1"

declare -a arr=(
  "Omon/hfds/gr/v20180701/hfds_Omon_GFDL-CM4_piControl_r1i1p1f1_gr_015101-017012.nc"
  "Ofx/areacello/gr/v20180701/areacello_Ofx_GFDL-CM4_piControl_r1i1p1f1_gr.nc"
  "Ofx/sftof/gr/v20180701/sftof_Ofx_GFDL-CM4_piControl_r1i1p1f1_gr.nc"
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
      outfile=${basename/015101-017012/015101-015103}
      cdo -selyear,151 $file $tmpfile > /dev/null 2>&1
      cdo -selmonth,1/3 $tmpfile $outfile > /dev/null 2>&1
      rm $tmpfile > /dev/null 2>&1
   fi
   echo "mkdir -p ${TEST_DIR}/${ROOT_DIR_TO_MAKE}/${dir_to_make}"
   echo "mv $outfile ${TEST_DIR}/${ROOT_DIR_TO_MAKE}/${dir_to_make}/${outfile}"
done
