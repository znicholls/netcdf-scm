"""
Experimental script to work on crunching a number of files to SCM format

Improvements in future:

- use CLI from Jared
- output datapackages (speak to Robert)
"""

from os import walk
from os.path import join


from netcdf_scm.iris_cube_wrappers import MarbleCMIP5Cube


INPUT_DIR = "tests/test_data/marble_cmip5"
OUTPUT_DIR = "."

for (dirpath, dirnames, filenames) in walk(INPUT_DIR):
    print(dirpath)
    print(dirnames)
    print(filenames)
    print(" ")
    if 'fx' in dirnames:
        dirnames.remove('fx')  # don't visit fx directories
    elif not dirnames:
        assert len(filenames) == 1
        import pdb
        pdb.set_trace()

