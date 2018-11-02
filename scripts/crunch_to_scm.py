from os import walk, makedirs, path
from os.path import join, isfile


from netcdf_scm.iris_cube_wrappers import MarbleCMIP5Cube
from progressbar import progressbar


INPUT_DIR = "../tests/test_data/marble_cmip5"
# INPUT_DIR = "/data/marble"
OUTPUT_DIR = "../output_examples/crunched_files"
# OUTPUT_DIR = "/data/marble/sandbox/znicholls/cmip5_crunched"
LAND_MASK_THRESHOLD = 50
VAR_TO_CRUNCH = "tas"

if not path.exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)


def crunch_data(
    in_dir, out_dir, var_to_crunch=None,
    land_mask_threshold=50, force_regeneration=False,
    output_prefix="scm_crunched",
):
    """Crunch data in a directory structure to OpenSCM csvs

    Crunched

    Parameters
    ----------
    in_dir : str
        Directory to walk to find files to crunch

    out_dir : str
        Directory in which to save output csvs

    var_to_crunch : str
        Variable to crunch. If None, crunch all variables.

    land_mask_threshold : float
        Land mask threshold to use when deciding which boxes are
        land and which are ocean in the input data.

    force_regeneration : bool
        If True, crunch file even if the output file already exists.

    output_prefix : str
        Prefix to attach to the input filenames when saving the
        crunched csvs.
    """
    print("Crunching:\n{}\n\n".format(in_dir))
    failures = []
    for (dirpath, dirnames, filenames) in progressbar(walk(in_dir)):
        if not dirnames:
            if (var_to_crunch is not None) and (var_to_crunch not in dirpath):
                continue
            try:
                scmcube = MarbleCMIP5Cube()
                if len(filenames) == 1:
                    out_filename = "{}_{}".format(
                        output_prefix,
                        filenames[0].replace(".nc", ".csv")
                    )
                    outfile = join(out_dir, out_filename)
                    if not force_regeneration and isfile(outfile):
                        continue
                    scmcube.load_data_from_path(join(dirpath, filenames[0]))
                else:
                    scmcube.load_data_in_directory(dirpath)
                    out_filename = "{}_{}".format(
                        output_prefix,
                        scmcube._get_data_filename().replace(".nc", ".csv")
                    )
                    outfile = join(out_dir, out_filename)
                    if not force_regeneration and isfile(outfile):
                        continue

                magicc_df = scmcube.get_scm_timeseries(
                    land_mask_threshold=land_mask_threshold
                )
                magicc_df.df = magicc_df.df.pivot_table(
                    values="value",
                    index=["time"],
                    columns=["variable", "unit", "region", "model", "scenario"],
                )

                magicc_df.df.to_csv(outfile)
            except Exception as exc:
                header = "Exception"
                exc_string = header + "\n" + "-" * len(header) + "\n" + str(exc)

                failures.append("{}\n{}\n{}".format(dirpath, filenames, exc_string))
                continue

    print("Failures\n========\n{}".format("\n\n".join(failures)))


crunch_data(
    INPUT_DIR, OUTPUT_DIR, var_to_crunch=VAR_TO_CRUNCH,
    land_mask_threshold=LAND_MASK_THRESHOLD, force_regeneration=True,
    output_prefix="scm_crunched",
)
