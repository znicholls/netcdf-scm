"""
Experimental script to work on crunching a number of files to SCM format

Improvements in future:

- use CLI from Jared
- output datapackages (speak to Robert)
"""

from os import walk, makedirs
from os import path
from os.path import join


from netcdf_scm.iris_cube_wrappers import MarbleCMIP5Cube
from progressbar import progressbar


INPUT_DIR = "tests/test_data/marble_cmip5"
OUTPUT_DIR = "crunched_files"
LAND_MASK_THRESHOLD = 50


if not path.exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)


for (dirpath, dirnames, filenames) in progressbar(walk(INPUT_DIR)):
    if 'fx' in dirnames:
        dirnames.remove('fx')  # don't visit fx directories
    elif not dirnames:
        assert len(filenames) == 1
        if not filenames[0].endswith(".nc"):
            continue

        scmcube = MarbleCMIP5Cube()
        scmcube.load_data_from_path(join(dirpath, filenames[0]))
        magicc_df = scmcube.get_scm_timeseries(land_mask_threshold=LAND_MASK_THRESHOLD)
        out_filename = "scm_crunched_{}".format(filenames[0].replace(".nc", ".csv"))
        magicc_df.df.to_csv(join(OUTPUT_DIR, out_filename), index=False)

