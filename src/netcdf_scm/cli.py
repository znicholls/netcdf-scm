"""Command line interface"""
import datetime as dt
import logging
import os
import os.path
import re
import sys
from os import makedirs, walk
from time import gmtime, strftime

import click
import numpy as np
import pymagicc
from openscm.scmdataframe import ScmDataFrame, df_append
from pymagicc.io import MAGICCData

from . import __version__
from .io import load_scmdataframe, save_netcdf_scm_nc
from .iris_cube_wrappers import (
    CMIP6Input4MIPsCube,
    CMIP6OutputCube,
    MarbleCMIP5Cube,
    SCMCube,
)
from .output import OutputFileDatabase
from .wranglers import convert_scmdf_to_tuningstruc

logger = logging.getLogger("netcdf_scm")

_CUBES = {
    "Scm": SCMCube,
    "MarbleCMIP5": MarbleCMIP5Cube,
    "CMIP6Input4MIPs": CMIP6Input4MIPsCube,
    "CMIP6Output": CMIP6OutputCube,
}

_MAGICC_VARIABLE_MAP = {"tas": ("Surface Temperature", "SURFACE_TEMP")}
"""Mapping from CMOR variable names to MAGICC variables"""


def init_logging(params, out_filename=None):
    """
    Set up the root logger

    All INFO messages and greater are written to stderr.
    If an ``out_filename`` is provided, all recorded log messages are also written to
    disk.

    # TODO: make level of logging customisable

    Parameters
    ----------
    params : list
        A list of key values to write at the start of the log

    out_filename : str
        Name of the log file which is written to disk
    """
    handlers = []
    if out_filename:
        h = logging.FileHandler(out_filename, "a")
        h.setLevel(logging.DEBUG)
        handlers.append(h)

    # Write logs to stderr
    h_stderr = logging.StreamHandler(sys.stderr)
    h_stderr.setLevel(logging.INFO)
    handlers.append(h_stderr)

    fmt = logging.Formatter("{asctime} {levelname}:{name}:{message}", style="{")
    for h in handlers:
        if h.formatter is None:
            h.setFormatter(fmt)
        logger.addHandler(h)

    # use DEBUG as default for now
    logger.setLevel(logging.DEBUG)

    logging.captureWarnings(True)
    logger.info("netcdf-scm: %s", __version__)
    for k, v in params:
        logger.info("%s: %s", k, v)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("src", type=click.Path(exists=True, readable=True, resolve_path=True))
@click.argument(
    "dst", type=click.Path(file_okay=False, writable=True, resolve_path=True)
)
@click.argument("crunch_contact")
@click.option(
    "--drs",
    default="Scm",
    type=click.Choice(["Scm", "MarbleCMIP5", "CMIP6Input4MIPs", "CMIP6Output"]),
    show_default=True,
    help="Data reference syntax to use for crunching.",
)
@click.option(
    "--regexp",
    default="^((?!fx).)*$",
    show_default=True,
    help="Regular expression to apply to file directory (only crunches matches).",
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
    "--force/--do-not-force",  # pylint:disable=too-many-arguments
    "-f",
    help="Overwrite any existing files.",
    default=False,
    show_default=True,
)
def crunch_data(
    src, dst, crunch_contact, drs, regexp, land_mask_threshold, data_sub_dir, force
):
    r"""
    Crunch data in ``src`` to NetCDF-SCM ``.nc`` files in ``dst``.

    ``src`` is searched recursively and netcdf-scm will attempt to crunch all the files
    found. The directory structure in ``src`` will be mirrored in ``dst``.

    Failures and warnings are recorded and written into a text file in ``dst``. We
    recommend examining this file using a file analysis tool such as ``grep``. We
    often use the command ``grep "\|WARNING\|INFO\|ERROR <log-file>``.

    ``crunch_contact`` is written into the output ``.nc`` files' ``crunch_contact``
    attribute.
    """
    output_prefix = "netcdf-scm"
    separator = "_"
    out_dir = os.path.join(dst, data_sub_dir)

    log_file = os.path.join(
        out_dir,
        "{}-crunch.log".format(_get_timestamp().replace(" ", "_").replace(":", "")),
    )
    _make_path_if_not_exists(out_dir)
    init_logging(
        [
            ("crunch-contact", crunch_contact),
            ("source", src),
            ("destination", out_dir),
            ("drs", drs),
            ("regexp", regexp),
            ("land_mask_threshold", land_mask_threshold),
            ("force", force),
        ],
        out_filename=log_file,
    )

    tracker = OutputFileDatabase(out_dir)
    
    #@profile
    def crunch_files(fnames, dpath):
        scmcube = _load_scm_cube(drs, dpath, fnames)

        out_filename = separator.join([output_prefix, scmcube.get_data_filename()])

        outfile_dir = scmcube.get_data_directory().replace(scmcube.root_dir, out_dir)
        out_filepath = os.path.join(outfile_dir, out_filename)

        _make_path_if_not_exists(outfile_dir)

        if not force and tracker.contains_file(out_filepath):
            logger.info("Skipped (already exists, not overwriting) %s", out_filepath)
            return

        results = scmcube.get_scm_timeseries_cubes(
            land_mask_threshold=land_mask_threshold
        )
        results = _set_crunch_contact_in_results(results, crunch_contact)

        tracker.register(out_filepath, scmcube.info)
        logger.info("Writing file to %s", out_filepath)
        save_netcdf_scm_nc(results, out_filepath)

    failures = _apply_func_to_files_if_dir_matches_regexp(
        crunch_files, src, re.compile(regexp)
    )

    if failures:
        raise click.ClickException(
            "Some files failed to process. See {} for more details".format(log_file)
        )


def _apply_func_to_files_if_dir_matches_regexp(apply_func, search_dir, regexp_to_match):
    failures = False

    logger.info("Finding directories with files")
    total_dirs = len(list([f for _, _, f in walk(search_dir) if f]))
    logger.info("Found %s directories with files", total_dirs)
    dir_counter = 1
    for dirpath, _, filenames in walk(search_dir):
        logger.debug("Entering %s", dirpath)
        if filenames:
            logger.info("Checking directory %s of %s", dir_counter, total_dirs)
            dir_counter += 1
            if not regexp_to_match.match(dirpath):
                logger.debug("Skipping (did not match regexp) %s", dirpath)
                continue
            logger.info("Attempting to process: %s", filenames)
            try:
                apply_func(filenames, dirpath)

            except Exception:  # pylint:disable=broad-except
                logger.exception("Failed to process: %s", filenames)
                failures = True

    return failures


def _load_scm_cube(drs, dirpath, filenames):
    scmcube = _get_scmcube_helper(drs)
    if len(filenames) == 1:
        scmcube.load_data_from_path(os.path.join(dirpath, filenames[0]))
    else:
        scmcube.load_data_in_directory(dirpath)

    return scmcube


def _set_crunch_contact_in_results(res, crunch_contact):
    for _, c in res.items():
        if "crunch_contact" in c.cube.attributes:
            logger.warning(
                "Overwriting `crunch_contact` attribute"
            )  # pragma: no cover # emergency valve
        c.cube.attributes["crunch_contact"] = crunch_contact

    return res


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("src", type=click.Path(exists=True, readable=True, resolve_path=True))
@click.argument(
    "dst", type=click.Path(file_okay=False, writable=True, resolve_path=True)
)
@click.argument("wrangle_contact")
@click.option(
    "--regexp",
    default="^((?!fx).)*$",
    show_default=True,
    help="Regular expression to apply to file directory (only wrangles matches).",
)
@click.option(
    "--prefix", default=None, help="Prefix to apply to output file names (not paths)."
)
@click.option(
    "--out-format",
    default="mag-files",
    type=click.Choice(
        [
            "mag-files",
            "magicc-input-files-point-end-of-year",
            "tuningstrucs-blend-model",
        ]
    ),
    show_default=True,
    help="Format to re-write csvs into.",
)
@click.option(
    "--drs",
    default="None",
    type=click.Choice(["None", "MarbleCMIP5", "CMIP6Input4MIPs", "CMIP6Output"]),
    show_default=True,
    help="Data reference syntax to use to decipher paths. This is required to ensure the output folders match the input data reference syntax.",
)
@click.option(
    "--force/--do-not-force",  # pylint:disable=too-many-arguments
    "-f",
    help="Overwrite any existing files.",
    default=False,
    show_default=True,
)
def wrangle_netcdf_scm_ncs(
    src, dst, wrangle_contact, regexp, prefix, out_format, drs, force
):
    """
    Wrangle NetCDF-SCM ``.nc`` files into other formats and directory structures.

    ``src`` is searched recursively and netcdf-scm will attempt to wrangle all the
    files found.

    ``wrangle_contact`` is written into the header of the output files.
    """
    log_file = os.path.join(
        dst,
        "{}-wrangle.log".format(_get_timestamp().replace(" ", "_").replace(":", "")),
    )
    _make_path_if_not_exists(dst)
    init_logging(
        [
            ("wrangle_contact", wrangle_contact),
            ("source", src),
            ("destination", dst),
            ("regexp", regexp),
            ("prefix", prefix),
            ("drs", drs),
            ("out_format", out_format),
            ("force", force),
        ],
        out_filename=log_file,
    )

    # TODO: turn all wranglers into subclasses of a base `Wrangler` class
    if out_format == "tuningstrucs-blend-model":
        _tuningstrucs_blended_model_wrangling(src, dst, regexp, force, drs, prefix)
    else:
        _do_wrangling(src, dst, regexp, out_format, force, wrangle_contact, drs)


def _tuningstrucs_blended_model_wrangling(  # pylint:disable=too-many-arguments
    src, dst, regexp, force, drs, prefix
):
    regexp_compiled = re.compile(regexp)
    considered_regexps = []
    for dirpath, _, filenames in walk(src):
        if filenames:
            if not regexp_compiled.match(dirpath):
                continue

            if considered_regexps:
                if any([r.match(dirpath) for r in considered_regexps]):
                    continue

            regexp_here = _get_blended_model_regexp(drs, dirpath)
            logger.info("Wrangling %s", regexp_here)

            _tuningstrucs_blended_model_wrangling_inner_loop(
                src, regexp_here, dst, force, prefix
            )
            considered_regexps.append(regexp_here)


def _get_blended_model_regexp(drs, dirpath):
    scmcube = _get_scmcube_helper(drs)
    ids = {
        k: v
        if any([s in k for s in ["variable", "experiment", "activity", "mip"]])
        else ".*"
        for k, v in scmcube.process_path(dirpath).items()
    }
    for name, value in ids.items():
        setattr(scmcube, name, value)

    return re.compile("{}.*".format(scmcube.get_data_directory()))


def _tuningstrucs_blended_model_wrangling_inner_loop(
    src, regexp_inner, dst, force, prefix
):
    collected = []
    for dirpath_inner, _, filenames_inner in walk(src):
        if filenames_inner:
            if not regexp_inner.match(dirpath_inner):
                continue

            openscmdf = df_append(
                [
                    load_scmdataframe(os.path.join(dirpath_inner, f))
                    for f in filenames_inner
                ]
            )
            tmp_ts = openscmdf.timeseries().reset_index()
            tmp_ts["unit"] = tmp_ts["unit"].astype(str)
            openscmdf = ScmDataFrame(tmp_ts)

            collected.append(openscmdf)

    convert_scmdf_to_tuningstruc(df_append(collected), dst, force=force, prefix=prefix)


def _do_wrangling(  # pylint:disable=too-many-arguments
    src, dst, regexp, out_format, force, wrangle_contact, drs
):
    regexp_compiled = re.compile(regexp)

    if out_format in ("mag-files", "magicc-input-files-point-end-of-year"):
        _do_magicc_wrangling(
            src, dst, regexp_compiled, out_format, force, wrangle_contact, drs
        )
    else:
        raise ValueError("Unsupported format: {}".format(out_format))


def _do_magicc_wrangling(  # pylint:disable=too-many-arguments
    src, dst, regexp_compiled, out_format, force, wrangle_contact, drs
):
    scmcube = _get_scmcube_helper(drs)

    def get_openscmdf_metadata_header(fnames, dpath):
        openscmdf = df_append(
            [load_scmdataframe(os.path.join(dpath, f)) for f in fnames]
        )
        metadata = openscmdf.metadata
        header = (
            "Date: {}\n"
            "Contact: {}\n"
            "Source data crunched with: NetCDF-SCM v{}\n"
            "File written with: pymagicc v{} (more info at "
            "github.com/openclimatedata/pymagicc)\n".format(
                dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                wrangle_contact,
                metadata["crunch_netcdf_scm_version"],
                pymagicc.__version__,
            )
        )

        return openscmdf, metadata, header

    def get_outfile_dir_symlink_dir(dpath):
        outfile_dir = dpath.replace(scmcube.process_path(dpath)["root_dir"], dst)
        _make_path_if_not_exists(outfile_dir)
        symlink_dir = os.path.join(dst, "flat")
        _make_path_if_not_exists(symlink_dir)

        return outfile_dir, symlink_dir

    if out_format == "mag-files":
        wrangle_to_mag_files = _get_wrangle_to_mag_files_func(
            force, get_openscmdf_metadata_header, get_outfile_dir_symlink_dir
        )

        failures = _apply_func_to_files_if_dir_matches_regexp(
            wrangle_to_mag_files, src, regexp_compiled
        )
        if failures:
            raise click.ClickException(
                "Some files failed to process. See the logs for more details"
            )

    elif out_format == "magicc-input-files-point-end-of-year":
        wrangle_to_magicc_input_files_point_end_of_year = _get_wrangle_to_magicc_input_files_point_end_of_year_func(
            force, get_openscmdf_metadata_header, get_outfile_dir_symlink_dir
        )

        failures = _apply_func_to_files_if_dir_matches_regexp(
            wrangle_to_magicc_input_files_point_end_of_year, src, regexp_compiled
        )
        if failures:
            raise click.ClickException(
                "Some files failed to process. See the logs for more details"
            )


def _get_wrangle_to_mag_files_func(
    force, get_openscmdf_metadata_header, get_outfile_dir_symlink_dir
):
    def wrangle_func(fnames, dpath):
        openscmdf, metadata, header = get_openscmdf_metadata_header(fnames, dpath)
        outfile_dir, symlink_dir = get_outfile_dir_symlink_dir(dpath)

        if len(fnames) > 1:
            raise AssertionError(
                "more than one file to wrangle?"
            )  # pragma: no cover # emergency valve

        out_file = os.path.join(outfile_dir, fnames[0])
        out_file = "{}.MAG".format(os.path.splitext(out_file)[0])

        if _skip_file(out_file, force, symlink_dir):
            return

        writer = MAGICCData(openscmdf)
        writer["todo"] = "SET"
        time_steps = writer.timeseries().columns[1:] - writer.timeseries().columns[:-1]
        step_upper = np.timedelta64(32, "D")  # pylint:disable=too-many-function-args
        step_lower = np.timedelta64(28, "D")  # pylint:disable=too-many-function-args
        if any((time_steps > step_upper) | (time_steps < step_lower)):
            raise ValueError(
                "Please raise an issue at "
                "github.com/znicholls/netcdf-scm/issues "
                "to discuss how to handle non-monthly data wrangling"
            )

        writer.metadata = metadata
        writer.metadata["timeseriestype"] = "MONTHLY"
        writer.metadata["header"] = header

        logger.info("Writing file to %s", out_file)
        writer.write(out_file, magicc_version=7)

        symlink_file = os.path.join(symlink_dir, os.path.basename(out_file))
        logger.info("Making symlink to %s", symlink_file)
        os.symlink(out_file, symlink_file)

    return wrangle_func


def _get_wrangle_to_magicc_input_files_point_end_of_year_func(
    force, get_openscmdf_metadata_header, get_outfile_dir_symlink_dir
):
    def wrangle_func(fnames, dpath):
        openscmdf, metadata, header = get_openscmdf_metadata_header(fnames, dpath)
        outfile_dir, symlink_dir = get_outfile_dir_symlink_dir(dpath)

        src_time_points = openscmdf.timeseries().columns
        out_time_points = [
            dt.datetime(y, 12, 31)
            for y in range(src_time_points[0].year, src_time_points[-1].year + 1)
        ]
        time_id = "{}-{}".format(src_time_points[0].year, src_time_points[-1].year + 1)
        try:
            openscmdf = openscmdf.interpolate(out_time_points)
        except (ValueError, AttributeError):
            logger.exception("Not happy %s", fnames)
            return

        _write_magicc_input_files(
            openscmdf, time_id, outfile_dir, symlink_dir, force, metadata, header
        )

    return wrangle_func


def _write_magicc_input_files(  # pylint:disable=too-many-arguments
    openscmdf, time_id, outfile_dir, symlink_dir, force, metadata, header
):
    try:
        var_to_write = openscmdf["variable"].unique()[0]
        variable_abbreviations = {
            "filename": var_to_write,
            "magicc_name": _MAGICC_VARIABLE_MAP[var_to_write][0],
            "magicc_internal_name": _MAGICC_VARIABLE_MAP[var_to_write][1],
        }
    except KeyError:
        logger.exception(
            "I don't know which MAGICC variable to use for input `%s`", var_to_write
        )
        return

    region_filters = {
        "FOURBOX": [
            "World|Northern Hemisphere|Land",
            "World|Southern Hemisphere|Land",
            "World|Northern Hemisphere|Ocean",
            "World|Southern Hemisphere|Ocean",
        ],
        "GLOBAL": ["World"],
    }
    for region_key, regions_to_keep in region_filters.items():
        out_file = os.path.join(
            outfile_dir,
            (
                ("{}_{}_{}_{}_{}_{}_{}.IN")
                .format(
                    variable_abbreviations["filename"],
                    openscmdf["scenario"].unique()[0],
                    openscmdf["climate_model"].unique()[0],
                    openscmdf["member_id"].unique()[0],
                    time_id,
                    region_key,
                    variable_abbreviations["magicc_internal_name"],
                )
                .upper()
            ),
        )
        symlink_file = os.path.join(symlink_dir, os.path.basename(out_file))

        if _skip_file(out_file, force, symlink_dir):
            return

        writer = MAGICCData(openscmdf).filter(region=regions_to_keep)
        writer["todo"] = "SET"
        writer["variable"] = variable_abbreviations["magicc_name"]
        writer.metadata = metadata
        writer.metadata["header"] = header
        writer.metadata["timeseriestype"] = "POINT_END_OF_YEAR"
        try:
            writer.write(out_file, magicc_version=7)
            logger.info("Making symlink to %s", symlink_file)
            os.symlink(out_file, symlink_file)
        except (ValueError, AttributeError):
            logger.exception("Not happy %s", out_file)


def _skip_file(out_file, force, symlink_dir):
    if not force and os.path.isfile(out_file):
        logger.info("Skipped (already exists, not overwriting) %s", out_file)
        return True

    if os.path.isfile(out_file):
        os.remove(out_file)
        os.remove(os.path.join(symlink_dir, os.path.basename(out_file)))

    return False


def _get_timestamp():
    return strftime("%Y%m%d %H%M%S", gmtime())


def _make_path_if_not_exists(path_to_check):
    if not os.path.exists(path_to_check):
        logger.info("Making output directory: %s", path_to_check)
        makedirs(path_to_check)


def _get_scmcube_helper(drs):
    if drs == "None":
        raise NotImplementedError(
            "`drs` == 'None' is not supported yet. Please raise an issue at "
            "github.com/znicholls/netcdf-scm/ with your use case if you need this "
            "feature."
        )

    return _CUBES[drs]()
