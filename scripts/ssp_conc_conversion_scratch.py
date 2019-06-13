import glob
import os

import iris

IN_DIR = "/data/marble/FrozenProject_Repository/CMIP6GHGConcentrationProjections_1_2_1"
IN_DIR_HIST = "/data/marble/FrozenProject_Repository/CMIP6GHGConcentrationHistorical_1_2_0"
OUT_DIR = "/data/marble/sandbox/share/cmip6-wrangled-ipsl-sandbox"

for f in glob.glob(os.path.join(IN_DIR_HIST, "*GMNHSH*2014.nc")):
    print(f)
    c = iris.load_cube(f)
    break
