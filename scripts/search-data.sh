#!/bin/bash
ROOT_DIR="/data/marble/cmip6/CMIP6/CMIP"

declare -a search_vars=(
  "tas"
  "tos"
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
   find ${ROOT_DIR} -name "*${i}*CESM2*historical_r10i1p1f1_gn*" -type f
done
