#!/bin/bash
ROOT_DIR="/data/marble/cmip6/CMIP6/CMIP"

declare -a search_vars=(
  "gpp"
  "cSoilFast"
  "tas"
  "hfds"
  "thetao"
  "areacella"
  "areacello"
  "volcello"
  "sftlf"
  "sftof"
)

## now loop through the above array
for i in "${search_vars[@]}"
do
   find ${ROOT_DIR} -name "*${i}*CESM2_*historical_r7i1p1f1*" -type f
done
