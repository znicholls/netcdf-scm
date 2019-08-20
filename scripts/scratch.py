from netcdf_scm.iris_cube_wrappers import CMIP6OutputCube

test  = CMIP6OutputCube()

import pdb
pdb.set_trace()
test.load_data_in_directory("/data/marble/cmip6/CMIP6/CMIP/NCAR/CESM2/historical/r10i1p1f1/Omon/tos/gn/v20190313/")
test.load_data_in_directory("tests/test-data/cmip6output/CMIP6/CMIP/NCAR/CESM2/historical/r10i1p1f1/Omon/tos/gn/v20190313")
