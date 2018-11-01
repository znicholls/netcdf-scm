"""
Experimental script to work on crunching a number of files to SCM format

Improvements in future:

- use CLI from Jared
- output datapackages (speak to Robert)
"""

from os import walk, makedirs
from os import path
from os.path import join, isfile


from netcdf_scm.iris_cube_wrappers import MarbleCMIP5Cube
from progressbar import progressbar


# INPUT_DIR = "/data/marble"
INPUT_DIR = "./tests/test_data/marble_cmip5"
# OUTPUT_DIR = "/data/marble/sandbox/znicholls/cmip5_crunched_files"
OUTPUT_DIR = "./output_examples/crunched_files"
LAND_MASK_THRESHOLD = 50
VAR_TO_CRUNCH = "tas"


if not path.exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)


failures = []
for (dirpath, dirnames, filenames) in progressbar(walk(INPUT_DIR)):
    if not dirnames:
        if VAR_TO_CRUNCH not in dirpath:
            continue
        try:
            scmcube = MarbleCMIP5Cube()
            if len(filenames) == 1:
                out_filename = "scm_crunched_{}".format(
                    filenames[0].replace(".nc", ".csv")
                )
                outfile = join(OUTPUT_DIR, out_filename)
                if isfile(outfile):
                    continue
                scmcube.load_data_from_path(join(dirpath, filenames[0]))
            else:
                if "Nor" in dirpath:
                    import pdb
                    pdb.set_trace()
                scmcube.load_data_in_directory(dirpath)
                out_filename = "scm_crunched_{}".format(scmcube._get_data_filename().replace(".nc", ".csv"))
                outfile = join(OUTPUT_DIR, out_filename)
                if isfile(outfile):
                    continue

            magicc_df = scmcube.get_scm_timeseries(
                land_mask_threshold=LAND_MASK_THRESHOLD
            )
            magicc_df.df = magicc_df.df.pivot_table(
                values="value",
                index=["time"],
                columns=["variable", "unit", "region", "model", "scenario"],
            )

            magicc_df.df.to_csv(outfile)
        except:
            failures.append("{}\n{}".format(dirpath, filenames))
            continue

print("Failures\n========\n{}".format("\n\n".join(failures)))
