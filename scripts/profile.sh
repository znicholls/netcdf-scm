#!/bin/bash

INPUT_DIR="/data/marble/cmip6/CMIP6/DCPP/IPSL/IPSL-CM6A-LR/dcppC-atl-control/r1i1p1f1/Amon/tas/gr/v20190110"
INPUT_DIR="tests/test-data/cmip6output/"

~/.local/bin/kernprof -l -v scripts/scratch.py ${INPUT_DIR} &> performance_tests_results/small_crunch_profile.txt


INPUT_DIR="/data/marble/cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r1i1p1f1/Amon/tas/gn/v20181126"

~/.local/bin/kernprof -l -v scripts/scratch.py ${INPUT_DIR} &> performance_tests_results/medium_crunch_profile.txt


INPUT_DIR="/data/marble/cmip6/CMIP6/CMIP/BCC/BCC-CSM2-MR/piControl/r1i1p1f1/Amon/tas/gn/v20181016"

#~/.local/bin/kernprof -l -v scripts/scratch.py ${INPUT_DIR} &> performance_tests_results/big_crunch_profile.txt
