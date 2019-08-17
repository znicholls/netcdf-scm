#!/bin/bash
ROOT_DIR="/data/marble/cmip5/historical/"

declare -a arr=(
  "Amon/tas/ACCESS1-0/r1i1p1/tas_Amon_ACCESS1-0_historical_r1i1p1_185001-200512.nc "
  "Omon/hfds/ACCESS1-0/r1i1p1/hfds_Omon_ACCESS1-0_historical_r1i1p1_185001-200512.nc"
  "fx/areacella/ACCESS1-0/r0i0p0/areacella_fx_ACCESS1-0_historical_r0i0p0.nc"
  "fx/areacello/ACCESS1-0/r0i0p0/areacello_fx_ACCESS1-0_historical_r0i0p0.nc"
  "fx/sftlf/ACCESS1-0/r0i0p0/sftlf_fx_ACCESS1-0_historical_r0i0p0.nc"
  "fx/sftof/ACCESS1-0/r0i0p0/sftof_fx_ACCESS1-0_historical_r0i0p0.nc"
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
      outfile=${basename/185001-200512/187701-187703}
      cdo -selyear,1877 $file $tmpfile > /dev/null 2>&1
      cdo -selmonth,1/3 $tmpfile $outfile > /dev/null 2>&1
      rm $tmpfile > /dev/null 2>&1
   fi
   echo "mkdir -p ${dir_to_make}"
   echo "mv $outfile ${dir_to_make}/${outfile}"
done
