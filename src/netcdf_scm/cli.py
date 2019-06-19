"""
Command line interface
"""
import datetime as dt
import logging
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

logger = logging.getLogger("netcdf-scm")

_CUBES = {
    "Scm": SCMCube,
    "MarbleCMIP5": MarbleCMIP5Cube,
    "CMIP6Input4MIPs": CMIP6Input4MIPsCube,
    "CMIP6Output": CMIP6OutputCube,
}

_MAGICC_VARIABLE_MAP = {"tas": ("Surface Temperature", "SURFACE_TEMP")}
"""Mapping from CMOR variable names to MAGICC variables"""


def init_logging(params, out_filename=None, level=None):
    """
    Set up the root logger

    All INFO messages and greater are written to stderr.
    If an ``out_filename`` is provided, all recorded log messages are also written to
    disk.

    Parameters
    ----------
    params : list
        A list of key values to write at the start of the log

    out_filename : str
        Name of the log file which is written to disk

    level : int
        If not `None`, sets the level of the root logger
    """
    handlers = []
    if out_filename:
        h = logging.FileHandler(out_filename, "a")
        h.setLevel(logging.DEBUG)
        handlers.append(h)

    # Write logs to stderr
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.INFO)
    handlers.append(h)

    root = logging.root
    fmt = logging.Formatter("{asctime} {levelname}:{name}:{message}", style="{")
    for h in handlers:
        if h.formatter is None:
            h.setFormatter(fmt)
        root.addHandler(h)

    if level is not None:
        root.setLevel(level)
    else:
        # root has to be lowest level to allow messages, let handlers deal with the
        # rest
        root.setLevel(logging.DEBUG)

    logging.captureWarnings(True)
    logger.info("netcdf-scm: {}".format(__version__))
    for k, v in params:
        logger.info("{}: {}".format(k, v))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("src", type=click.Path(exists=True, readable=True, resolve_path=True))
@click.argument(
    "dst", type=click.Path(file_okay=False, writable=True, resolve_path=True)
)
@click.argument("crunch_contact")
@click.option(
    "--cube-type",
    default="Scm",
    type=click.Choice(["Scm", "MarbleCMIP5", "CMIP6Input4MIPs", "CMIP6Output"]),
    show_default=True,
    help="Cube to use for crunching.",
)
@click.option(
    "--regexp",
    default="^((?!fx).)*$",
    show_default=True,
    help="Regular expression to apply to filepath (only crunches matches).",
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
    src,
    dst,
    crunch_contact,
    cube_type,
    regexp,
    land_mask_threshold,
    data_sub_dir,
    force,
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
    timestamp = _get_timestamp()
    out_dir = os.path.join(dst, data_sub_dir)

    log_params = [
        ("crunch-contact", crunch_contact),
        ("source", src),
        ("destination", out_dir),
        ("cube-type", cube_type),
        ("regexp", regexp),
        ("land_mask_threshold", land_mask_threshold),
        ("force", force),
    ]
    log_file = os.path.join(
        out_dir, "{}-crunch.log".format(timestamp.replace(" ", "_").replace(":", ""))
    )
    _make_path_if_not_exists(out_dir)
    init_logging(log_params, out_filename=log_file)

    failures = False

    regexp_compiled = re.compile(regexp)

    tracker = OutputFileDatabase(out_dir)
    logger.info("Finding directories with files")
    total_dirs = len(list([f for _, _, f in walk(src) if f]))
    logger.info("Found {} directories with files".format(total_dirs))
    dir_counter = 1
    for dirpath, _, filenames in walk(src):
        logger.debug("Entering {}".format(dirpath))
        if filenames:
            logger.info("Checking directory {} of {}".format(dir_counter, total_dirs))
            dir_counter += 1
            if not regexp_compiled.match(dirpath):
                logger.debug("Skipping (did not match regexp) {}".format(dirpath))
                continue
            logger.info("Attempting to process: {}".format(filenames))
            scmcube = _CUBES[cube_type]()
            try:
                if len(filenames) == 1:
                    scmcube.load_data_from_path(os.path.join(dirpath, filenames[0]))
                else:
                    scmcube.load_data_in_directory(dirpath)

                out_filename = separator.join(
                    [output_prefix, scmcube._get_data_filename()]
                )

                out_filedir = scmcube._get_data_directory().replace(
                    scmcube.root_dir, out_dir
                )
                out_filepath = os.path.join(out_filedir, out_filename)

                _make_path_if_not_exists(out_filedir)

                if not force and tracker.contains_file(out_filepath):
                    logger.info(
                        "Skipped (already exists, not overwriting) {}".format(
                            out_filepath
                        )
                    )
                    continue

                results = scmcube.get_scm_timeseries_cubes(
                    land_mask_threshold=land_mask_threshold
                )
                for _, c in results.items():
                    if "crunch_contact" in c.cube.attributes:
                        logger.warning(
                            "Overwriting `crunch_contact` attribute"
                        )  # pragma: no cover # emergency valve
                    c.cube.attributes["crunch_contact"] = crunch_contact

                tracker.register(out_filepath, scmcube.info)
                save_netcdf_scm_nc(results, out_filepath)

            except Exception:
                logger.exception("Failed to process: {}".format(filenames))
                failures = True

    if failures:
        raise click.ClickException(
            "Some files failed to process. See {} for more details".format(log_file)
        )


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
    help="Regular expression to apply to filepath (only wrangles matches).",
)
@click.option(
    "--nested/--flat",
    help="Maintain source directory structure in destination. If `flat`, writes all files to a single directory.",
    default=True,
    show_default=True,
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
    help="Data reference syntax to use to decipher paths when crunching to flat and the output format is tuningstrucs. This is required to ensure the output names are unique.",
)
@click.option(
    "--force/--do-not-force",
    "-f",
    help="Overwrite any existing files.",
    default=False,
    show_default=True,
)
def wrangle_netcdf_scm_ncs(
    src, dst, wrangle_contact, regexp, nested, prefix, out_format, drs, force
):
    """
    Wrangle NetCDF-SCM ``.nc`` files into other formats and directory structures.

    ``src`` is searched recursively and netcdf-scm will attempt to wrangle all the
    files found.

    ``wrangle_contact`` is written into the header of the output files.
    """
    if out_format == "tuningstrucs-blend-model" and nested:
        raise ValueError("Cannot wrangle to nested tuningstrucs with blended models")

    log_params = [
        ("wrangle_contact", wrangle_contact),
        ("source", src),
        ("destination", dst),
        ("regexp", regexp),
        ("land_mask_threshold", nested),
        ("drs", drs),
        ("out_format", out_format),
        ("force", force),
    ]
    log_file = os.path.join(
        dst,
        "{}-wrangle.log".format(_get_timestamp().replace(" ", "_").replace(":", "")),
    )
    _make_path_if_not_exists(dst)
    init_logging(log_params, out_filename=log_file)

    if out_format == "tuningstrucs-blend-model":
        _tuningstrucs_blended_model_wrangling(src, dst, regexp, force, drs, prefix)
    else:
        _do_wrangling(
            src, dst, regexp, nested, out_format, force, prefix, wrangle_contact
        )


def _tuningstrucs_blended_model_wrangling(src, dst, regexp, force, drs, prefix):
    regexp_compiled = re.compile(regexp)
    considered_regexps = []
    for dirpath, _, filenames in walk(src):
        if filenames:
            if not regexp_compiled.match(dirpath):
                continue

            if considered_regexps:
                if any([r.match(dirpath) for r in considered_regexps]):
                    continue

            if drs == "None":
                raise NotImplementedError(
                    "`drs` == 'None' is not supported for wrangling to "
                    "tuningstrucs. Please raise an issue at "
                    "github.com/znicholls/netcdf-scm/ if you need this feature."
                )

            scmcube = _CUBES[drs]()
            ids = {
                k: v
                if any([s in k for s in ["variable", "experiment", "activity", "mip"]])
                else ".*"
                for k, v in scmcube.process_path(dirpath).items()
            }

            regexp_here = re.compile(
                os.path.dirname(
                    scmcube.get_filepath_from_load_data_from_identifiers_args(**ids)
                )
            )
            logger.info("Wrangling {}".format(regexp_here))
            regexp_compiled = re.compile(regexp)

            collected = []
            for dirpath_inner, _, filenames_inner in walk(src):
                if filenames_inner:
                    if not regexp_compiled.match(dirpath_inner):
                        continue

                    openscmdf = df_append(
                        [load_scmdataframe(os.path.join(dirpath_inner, f)) for f in filenames_inner]
                    )
                    tmp_ts = openscmdf.timeseries().reset_index()
                    tmp_ts["unit"] = tmp_ts["unit"].astype(str)
                    openscmdf = ScmDataFrame(tmp_ts)

                    collected.append(openscmdf)

            logger.info("Wrangling {}".format(regexp_here))
            convert_scmdf_to_tuningstruc(
                df_append(collected), dst, force=force, prefix=prefix
            )

            considered_regexps.append(regexp_here)


def _do_wrangling(src, dst, regexp, nested, out_format, force, prefix, wrangle_contact):
    regexp_compiled = re.compile(regexp)

    logger.info("Finding directories with files")
    total_dirs = len([f for _, _, f in walk(src) if f])
    logger.info("Found {} directories with files".format(total_dirs))
    dir_counter = 1
    for dirpath, _, filenames in walk(src):
        if filenames:
            logger.info("Checking directory {} of {}".format(dir_counter, total_dirs))
            dir_counter += 1
            if not regexp_compiled.match(dirpath):
                logger.debug("Skipping (did not match regexp) {}".format(dirpath))
                continue

            openscmdf = df_append(
                [load_scmdataframe(os.path.join(dirpath, f)) for f in filenames]
            )
            metadata = openscmdf.metadata
            tmp_ts = openscmdf.timeseries().reset_index()
            tmp_ts["unit"] = tmp_ts["unit"].astype(str)
            openscmdf = ScmDataFrame(tmp_ts)

            out_filedir = dirpath.replace(src, dst) if nested else dst

            header = (
                "Date: {}\n"
                "Contact: {}\n"
                "Source data crunched with: NetCDF-SCM v{}\n"
                "File written with: pymagicc v{} (more info at github.com/openclimatedata/pymagicc)\n".format(
                    dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    wrangle_contact,
                    metadata["crunch_netcdf_scm_version"],
                    pymagicc.__version__,
                )
            )

            if out_format == "mag-files":
                assert len(filenames) == 1, "more than one file to wrangle?"
                _make_path_if_not_exists(out_filedir)
                out_file = os.path.join(out_filedir, filenames[0])
                out_file = "{}.MAG".format(os.path.splitext(out_file)[0])
                if not force and os.path.isfile(out_file):
                    logger.info(
                        "Skipped (already exists, not overwriting) {}".format(out_file)
                    )
                    continue

                writer = MAGICCData(openscmdf)
                writer["todo"] = "SET"
                time_steps = (
                    writer.timeseries().columns[1:] - writer.timeseries().columns[:-1]
                )
                if any(
                    (time_steps > np.timedelta64(32, "D"))
                    | (time_steps < np.timedelta64(28, "D"))
                ):
                    raise ValueError(
                        "Please raise an issue at github.com/znicholls/netcdf-scm/"
                        "issues to discuss how to handle non-monthly data wrangling"
                    )
                writer.metadata = metadata
                writer.metadata["timeseriestype"] = "MONTHLY"
                writer.metadata["header"] = header
                writer.write(out_file, magicc_version=7)
            elif out_format == "magicc-input-files-point-end-of-year":
                src_time_points = openscmdf.timeseries().columns
                out_time_points = [
                    dt.datetime(y, 12, 31)
                    for y in range(
                        src_time_points[0].year, src_time_points[-1].year + 1
                    )
                ]
                time_id = "{}-{}".format(
                    src_time_points[0].year, src_time_points[-1].year + 1
                )
                openscmdf = openscmdf.interpolate(out_time_points)

                var_to_write = openscmdf["variable"].unique()[0]
                try:
                    magicc_var, magicc_in_file_var = _MAGICC_VARIABLE_MAP[var_to_write]
                except KeyError:
                    logger.exception(
                        "I don't know which MAGICC variable to use for input "
                        "`{}`".format(var_to_write)
                    )
                    continue

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
                    out_name = (
                        ("{}_{}_{}_{}_{}_{}_{}.IN")
                        .format(
                            var_to_write,
                            openscmdf["scenario"].unique()[0],
                            openscmdf["climate_model"].unique()[0],
                            openscmdf["member_id"].unique()[0],
                            time_id,
                            region_key,
                            magicc_in_file_var,
                        )
                        .upper()
                    )

                    out_file = os.path.join(out_filedir, out_name)
                    _make_path_if_not_exists(out_filedir)
                    if not force and os.path.isfile(out_file):
                        logger.info(
                            "Skipped (already exists, not overwriting) {}".format(
                                out_file
                            )
                        )
                        continue

                    writer = MAGICCData(openscmdf).filter(region=regions_to_keep)
                    writer["todo"] = "SET"
                    writer["variable"] = magicc_var
                    writer.metadata = metadata
                    writer.metadata["header"] = header
                    writer.metadata["timeseriestype"] = "POINT_END_OF_YEAR"
                    writer.write(out_file, magicc_version=7)
            else:
                raise ValueError("Unsupported format: {}".format(out_format))


def _get_timestamp():
    return strftime("%Y%m%d %H%M%S", gmtime())


def _make_path_if_not_exists(path_to_check):
    if not os.path.exists(path_to_check):
        logger.info("Making output directory: {}".format(path_to_check))
        makedirs(path_to_check)
