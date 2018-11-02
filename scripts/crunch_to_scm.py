from os import walk, makedirs, path
from os.path import join, isfile
import warnings
import time


from netcdf_scm.iris_cube_wrappers import MarbleCMIP5Cube
import progressbar


INPUT_DIR = "./tests/test_data/marble_cmip5"
OUTPUT_DIR = "./output_examples/crunched_files"
LAND_MASK_THRESHOLD = 50
VAR_TO_CRUNCH = "tas"

if not path.exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)


def crunch_data(
    in_dir,
    out_dir,
    var_to_crunch=None,
    land_mask_threshold=50,
    force_regeneration=False,
    output_prefix="scm_crunched",
):
    """Crunch data in a directory structure to OpenSCM csvs

    Failures are written into a text file in the directory above
    ``out_dir``

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
    print("Crunching:\n{}\n\nto\n{}".format(in_dir, out_dir))
    time.sleep(0.5)  # needed to get logging bar in right place...
    # really should use a logger here
    with warnings.catch_warnings(record=True) as recorded_warns:
        failures = []
        format_custom_text = progressbar.FormatCustomText(
            "Current directory :: %(curr_dir)-400s",
            {"curr_dir": "uninitialised"},
        )
        bar = progressbar.ProgressBar(
            widgets=[progressbar.SimpleProgress(), ". ", format_custom_text],
            max_value=len([w for w in walk(in_dir)]),
            prefix="Visiting directory ",
        ).start()
        for i, (dirpath, dirnames, filenames) in enumerate(walk(in_dir)):
            format_custom_text.update_mapping(curr_dir=dirpath)
            bar.update(i)
            if not dirnames:
                if (var_to_crunch is not None) and (var_to_crunch not in dirpath):
                    continue
                try:
                    scmcube = MarbleCMIP5Cube()
                    if len(filenames) == 1:
                        out_filename = "{}_{}".format(
                            output_prefix, filenames[0].replace(".nc", ".csv")
                        )
                        outfile = join(out_dir, out_filename)
                        if not force_regeneration and isfile(outfile):
                            continue
                        scmcube.load_data_from_path(join(dirpath, filenames[0]))
                    else:
                        scmcube.load_data_in_directory(dirpath)
                        out_filename = "{}_{}".format(
                            output_prefix,
                            scmcube._get_data_filename().replace(".nc", ".csv"),
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

                    # ideally would write to a logger here
                    failures.append("{}\n{}\n{}".format(dirpath, filenames, exc_string))
                    continue
        bar.finish()

    header_underline = "========"
    msg_underline = "--------"
    warnings_together = "\n\n{}\n\n".format(msg_underline).join([str(rw.message) for rw in recorded_warns])
    warnings_string = "Warnings\n{}\n{}".format(header_underline, warnings_together)
    failures_string = "Failures\n========\n{}".format(msg_underline.join(failures))
    output_string = "{}\n\n{}".format(failures_string, warnings_string)
    print(output_string)
    with open(join(OUTPUT_DIR, "..", "{}_failures_and_warnings.txt".format(output_prefix)), "w") as ef:
        ef.write(output_string)


crunch_data(
    INPUT_DIR,
    OUTPUT_DIR,
    var_to_crunch=VAR_TO_CRUNCH,
    land_mask_threshold=LAND_MASK_THRESHOLD,
    force_regeneration=True,
    output_prefix="scm_crunched",
)
