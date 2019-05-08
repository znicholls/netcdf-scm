"""Command line interface
"""
from os import walk, makedirs, path
from os.path import join, isfile
import re
import warnings
import traceback
import time
from time import gmtime, strftime

import click
import netcdf_scm
from netcdf_scm.iris_cube_wrappers import (
    SCMCube,
    MarbleCMIP5Cube,
    CMIP6Input4MIPsCube,
    CMIP6OutputCube,
)
import progressbar


_CUBES = {
    "Scm": SCMCube,
    "MarbleCMIP5": MarbleCMIP5Cube,
    "CMIP6Input4MIPs": CMIP6Input4MIPsCube,
    "CMIP6Output": CMIP6OutputCube,
}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("src", type=click.Path(exists=True, readable=True, resolve_path=True))
@click.argument(
    "dst", type=click.Path(file_okay=False, writable=True, resolve_path=True)
)
@click.option(
    "--cube-type",
    default="Scm",
    type=click.Choice(["Scm", "MarbleCMIP5", "CMIP6Input4MIPs", "CMIP6Output"]),
    show_default=True,
    help="Cube to use for crunching.",
)
@click.option(
    "--var-to-crunch",
    default=".*",
    show_default=True,
    help="Variable to crunch (uses regexp syntax, matches on filepaths).",
)
@click.option(
    "--land-mask-threshold",
    default=50.0,
    show_default=True,
    help="Minimum land fraction for a box to be considered land.",
)
@click.option(
    "--data-sub-dir",
    default="netcdf-scm-crunched",
    show_default=True,
    help="Sub-directory of ``dst`` to save data in.",
)
@click.option(
    "--force/--do-not-force",
    "-f",
    help="Overwrite any existing files.",
    default=False,
    show_default=True,
)
def crunch_data(
    src, dst, cube_type, var_to_crunch, land_mask_threshold, data_sub_dir, force
):
    """Crunch data in ``src`` to OpenSCM csv's in ``dst``.

    ``src`` is searched recursively and netcdf-scm will attemp to crunch all the files
    found. The directory structure in ``src`` will be mirrored in ``dst``.

    Failures and warnings are recorded and written into a text file in ``dst``.
    """
    title = "NetCDF Crunching"
    output_prefix = "netcdf-scm"
    separator = "_"
    timestamp = strftime("%Y%m%d %H%M%S", gmtime())
    out_dir = join(dst, data_sub_dir)

    metadata_header = (
        "{}\n"
        "{}\n"
        "NetCDF SCM version: {}\n"
        "\n"
        "time: {}\n"
        "cube-type: {}\n"
        "source: {}\n"
        "destination: {}\n"
        "var-to-crunch: {}\n"
        "land-mask-threshold: {}\n"
        "force: {}\n\n"
        "".format(
            title,
            "=" * len(title),
            netcdf_scm.__version__,
            timestamp,
            cube_type,
            src,
            out_dir,
            var_to_crunch,
            land_mask_threshold,
            force,
        )
    )
    click.echo(metadata_header)

    if not path.exists(out_dir):
        click.echo("Making output directory: {}\n".format(out_dir))
        makedirs(out_dir)

    already_exist_files = []

    var_regexp = re.compile(var_to_crunch)

    time.sleep(0.5)  # needed to get logging bar in right place...
    # really should use a logger here
    with warnings.catch_warnings(record=True) as recorded_warns:
        failures = []
        format_custom_text = progressbar.FormatCustomText(
            "Current directory :: %(curr_dir)-400s", {"curr_dir": "uninitialised"}
        )
        bar = progressbar.ProgressBar(
            widgets=[progressbar.SimpleProgress(), ". ", format_custom_text],
            max_value=len([w for w in walk(src)]),
            prefix="Visiting directory ",
        ).start()
        for i, (dirpath, dirnames, filenames) in enumerate(walk(src)):
            if not dirnames:
                if not var_regexp.match(dirpath):
                    continue
                format_custom_text.update_mapping(curr_dir=dirpath)
                bar.update(i)
                scmcube = _CUBES[cube_type]()
                try:
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
                        if not force and isfile(out_filepath):
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

                        if not force and isfile(out_filepath):
                            already_exist_files.append(out_filepath)
                            continue

                    results = scmcube.get_scm_timeseries(
                        land_mask_threshold=land_mask_threshold
                    )
                    results.to_csv(out_filepath)

                except Exception:
                    header = "Exception"
                    exc_string = (
                        header
                        + "\n"
                        + "-" * len(header)
                        + "\n"
                        + traceback.format_exc()
                    )

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

    already_exist_header = "Skipped (already exist, not overwriting)"
    if already_exist_files:
        already_exist_files_string = "- {}\n".format("\n- ".join(already_exist_files))
    else:
        already_exist_files_string = ""
    already_exist_string = "{}\n{}\n{}".format(
        already_exist_header,
        len(already_exist_header) * header_underline,
        already_exist_files_string,
    )

    output_string = "\n\n{}\n\n{}\n\n{}\n\n{}".format(
        metadata_header, failures_string, warnings_string, already_exist_string
    )
    click.echo(output_string)
    summary_file = join(
        out_dir,
        "{}-failures-and-warnings.txt".format(
            timestamp.replace(" ", "_").replace(":", "")
        ),
    )
    with open(summary_file, "w") as ef:
        ef.write(output_string)

    if failures:
        raise click.ClickException("Failures were found")
