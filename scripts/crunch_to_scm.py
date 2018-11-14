import sys
import argparse
import os
from os import walk, makedirs, path
from os.path import join, isfile
import warnings
import time
from time import gmtime, strftime


import netcdf_scm
from netcdf_scm.iris_cube_wrappers import MarbleCMIP5Cube
import progressbar


def crunch_data(
    in_dir,
    out_dir,
    var_to_crunch=None,
    land_mask_threshold=50,
    force_regeneration=False,
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
    output_prefix = "netcdf-scm"
    separator = "_"
    timestamp = strftime("%Y%m%d %H%M%S", gmtime())
    out_sub_dir = "netcdf-scm-crunched"
    out_dir = join(out_dir, out_sub_dir)
    print("\nCrunching:\n{}\n\nto\n{}\n".format(in_dir, out_dir))

    if not path.exists(out_dir):
        print("Making output directory: {}\n".format(out_dir))
        makedirs(out_dir)

    already_exist_files = []

    time.sleep(0.5)  # needed to get logging bar in right place...
    # really should use a logger here
    with warnings.catch_warnings(record=True) as recorded_warns:
        failures = []
        format_custom_text = progressbar.FormatCustomText(
            "Current directory :: %(curr_dir)-400s", {"curr_dir": "uninitialised"}
        )
        bar = progressbar.ProgressBar(
            widgets=[progressbar.SimpleProgress(), ". ", format_custom_text],
            max_value=len([w for w in walk(in_dir)]),
            prefix="Visiting directory ",
        ).start()
        for i, (dirpath, dirnames, filenames) in enumerate(walk(in_dir)):
            if not dirnames:
                if (var_to_crunch is not None) and (
                    var_to_crunch + os.sep not in dirpath
                ):
                    continue
                format_custom_text.update_mapping(curr_dir=dirpath)
                bar.update(i)
                try:
                    # todo: should be an arg/autodiscovered
                    scmcube = MarbleCMIP5Cube()
                    if len(filenames) == 1:
                        out_filename = separator.join(
                            [output_prefix, filenames[0].replace(".nc", ".csv")]
                        )
                        scmcube.load_data_from_path(join(dirpath, filenames[0]))
                        out_filedir = scmcube._get_data_directory().replace(
                            scmcube.root_dir, out_dir
                        )
                        out_filepath = join(out_filedir, out_filename)
                        if not path.exists(out_filedir):
                            makedirs(out_filedir)
                        if not force_regeneration and isfile(out_filepath):
                            already_exist_files.append(out_filepath)
                            continue
                    else:
                        scmcube.load_data_in_directory(dirpath)
                        out_filename = separator.join(
                            [
                                output_prefix,
                                scmcube._get_data_filename().replace(".nc", ".csv"),
                            ]
                        )

                        out_filedir = scmcube._get_data_directory().replace(
                            scmcube.root_dir, out_dir
                        )
                        out_filepath = join(out_filedir, out_filename)
                        if not path.exists(out_filedir):
                            makedirs(out_filedir)

                        if not force_regeneration and isfile(out_filepath):
                            already_exist_files.append(out_filepath)
                            continue

                    magicc_df = scmcube.get_scm_timeseries(
                        land_mask_threshold=land_mask_threshold
                    )
                    magicc_df.df = magicc_df.df.pivot_table(
                        values="value",
                        index=["time"],
                        columns=["variable", "unit", "region", "model", "scenario"],
                    )
                    magicc_df.df.to_csv(out_filepath)

                except Exception as exc:
                    header = "Exception"
                    exc_string = header + "\n" + "-" * len(header) + "\n" + str(exc)

                    # ideally would write to a logger here
                    failures.append("{}\n{}\n{}".format(dirpath, filenames, exc_string))
                    continue
        bar.finish()

    header_underline = "="
    msg_underline = "-"

    warnings_together = "\n\n{}\n\n".format(15 * msg_underline).join(
        [str(rw.message) for rw in recorded_warns]
    )

    warnings_header = "Warnings"
    warnings_string = "{}\n{}\n{}".format(
        warnings_header, len(warnings_header) * header_underline, warnings_together
    )

    failures_header = "Failures"
    failures_string = "{}\n{}\n{}".format(
        failures_header,
        len(failures_header) * header_underline,
        msg_underline.join(failures),
    )

    already_exist_header = "Already exist"
    if already_exist_files:
        already_exist_files_string = "- {}\n".format("\n- ".join(already_exist_files))
    else:
        already_exist_files_string = ""
    already_exist_string = "{}\n{}\n{}".format(
        already_exist_header,
        len(already_exist_header) * header_underline,
        already_exist_files_string,
    )

    metadata_header = (
        "Files crunched with NetCDF SCM\ntimestamp: {}\n\n"
        "NetCDF SCM version: {}\n"
        "input: {}\n"
        "output: {}\n"
        "var-to-crunch: {}\n"
        "land-mask-threshold: {}\n"
        "force: {}\n".format(
            timestamp,
            netcdf_scm.__version__,
            in_dir,
            out_dir,
            var_to_crunch,
            land_mask_threshold,
            force_regeneration,
        )
    )
    output_string = "{}\n\n{}\n\n{}\n\n{}".format(
        metadata_header, failures_string, warnings_string, already_exist_string
    )
    print(output_string)
    summary_file = join(
        out_dir, separator.join([out_sub_dir, "failures-and-warnings.txt"])
    )
    with open(summary_file, "w") as ef:
        ef.write(output_string)


def main():
    parser = argparse.ArgumentParser(
        prog="crunch-to-netcdf-scm",
        description="Crunch netCDF files to NetCDF SCM csv files",
    )

    # Command line args
    parser.add_argument("input", help="Root folder of the data to crunch.")
    parser.add_argument("output", help="Root folder in which to save the output data")
    parser.add_argument("--var-to-crunch", help="Variable to crunch", default=None)
    parser.add_argument(
        "--land-mask-threshold",
        help="Minimum fraction of a box which must be land for it to be counted as a land box",
        default=50,
        type=float,
    )
    parser.add_argument(
        "-f", "--force", help="Overwrite any existing files", action="store_true"
    )

    args = parser.parse_args()

    crunch_data(
        args.input,
        args.output,
        var_to_crunch=args.var_to_crunch,
        land_mask_threshold=args.land_mask_threshold,
        force_regeneration=args.force,
    )


if __name__ == "__main__":
    main()
