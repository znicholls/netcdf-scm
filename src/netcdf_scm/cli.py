"""Command line interface
"""
import logging
import re
import sys
from os import makedirs, path, walk
from os.path import dirname, join
from time import gmtime, strftime

import click
import progressbar
from openscm.scmdataframe import ScmDataFrame, df_append

import netcdf_scm

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


def init_logging(params, out_filename=None, level=None):
    """
    Set up the root logger

    All INFO messages and greater are written to stderr
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

    logging.captureWarnings(True)
    logger.info("netcdf-scm: {}".format(netcdf_scm.__version__))
    for k, v in params:
        logger.info("{}: {}".format(k, v))


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
def crunch_data(src, dst, cube_type, regexp, land_mask_threshold, data_sub_dir, force):
    """
    Crunch data in ``src`` to OpenSCM csv's in ``dst``.

    ``src`` is searched recursively and netcdf-scm will attempt to crunch all the files
    found. The directory structure in ``src`` will be mirrored in ``dst``.

    Failures and warnings are recorded and written into a text file in ``dst``.
    """

    output_prefix = "netcdf-scm"
    separator = "_"
    timestamp = _get_timestamp()
    out_dir = join(dst, data_sub_dir)

    log_params = [
        ("cube-type", cube_type),
        ("source", src),
        ("destination", out_dir),
        ("regexp", regexp),
        ("land_mask_threshold", land_mask_threshold),
        ("force", force),
    ]
    log_file = join(
        out_dir, "{}-crunch.log".format(timestamp.replace(" ", "_").replace(":", ""))
    )
    _make_path_if_not_exists(out_dir)
    init_logging(log_params, out_filename=log_file)

    failures = False

    regexp_compiled = re.compile(regexp)

    format_custom_text = _get_format_custom_text()
    bar = _get_progressbar(
        text=format_custom_text, max_value=len([w for w in walk(src)])
    )
    tracker = OutputFileDatabase(out_dir)
    for i, (dirpath, dirnames, filenames) in enumerate(walk(src)):
        logger.debug("Entering {}".format(dirpath))
        if filenames:
            if not regexp_compiled.match(dirpath):
                continue
            logger.info("Attempting to process: {}".format(filenames))
            format_custom_text.update_mapping(curr_dir=dirpath)
            bar.update(i)
            scmcube = _CUBES[cube_type]()
            try:
                if len(filenames) == 1:
                    out_filename = separator.join(
                        [output_prefix, filenames[0].replace(".nc", ".csv")]
                    )
                    scmcube.load_data_from_path(join(dirpath, filenames[0]))

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

                _make_path_if_not_exists(out_filedir)

                if not force and tracker.contains_file(out_filepath):
                    logger.info(
                        "Skipped (already exists, not overwriting) {}".format(
                            out_filepath
                        )
                    )
                    continue
                results = scmcube.get_scm_timeseries(
                    land_mask_threshold=land_mask_threshold
                )
                tracker.register(out_filepath, scmcube.info)
                results.to_csv(out_filepath)

            except Exception:
                logger.exception("Failed to process: {}".format(filenames))
                failures = True

        bar.finish()

    if failures:
        raise click.ClickException(
            "Some files failed to process. See {} for more details".format(log_file)
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("src", type=click.Path(exists=True, readable=True, resolve_path=True))
@click.argument(
    "dst", type=click.Path(file_okay=False, writable=True, resolve_path=True)
)
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
    "--prefix",
    default=None,
    help="Prefix to apply to output file names (not paths).",
)
@click.option(
    "--out-format",
    default="tuningstrucs",
    type=click.Choice(["tuningstrucs", "tuningstrucs-blend-model"]),
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
def wrangle_openscm_csvs(src, dst, regexp, nested, prefix, out_format, drs, force):
    """
    Wrangle OpenSCM csv files into other formats and directory structures

    ``src`` is searched recursively and netcdf-scm will attemp to wrangle all the files
    found.
    """
    if out_format == "tuningstrucs-blend-model" and nested:
        raise ValueError("Cannot wrangle to nested tuningstrucs with blended models")

    log_params = [
        ("source", src),
        ("destination", dst),
        ("regexp", regexp),
        ("land_mask_threshold", nested),
        ("drs", drs),
        ("out_format", out_format),
        ("force", force),
    ]
    log_file = join(
        dst,
        "{}-wrangle.log".format(_get_timestamp().replace(" ", "_").replace(":", "")),
    )
    _make_path_if_not_exists(dst)
    init_logging(log_params, out_filename=log_file)

    if out_format == "tuningstrucs-blend-model":
        _tuningstrucs_blended_model_wrangling(src, dst, regexp, force, drs, prefix)
    else:
        _do_wrangling(src, dst, regexp, nested, out_format, force, prefix)


def _tuningstrucs_blended_model_wrangling(src, dst, regexp, force, drs, prefix):
    regexp_compiled = re.compile(regexp)
    considered_regexps = []
    for i, (dirpath, dirnames, filenames) in enumerate(walk(src)):
        if filenames:
            if not regexp_compiled.match(dirpath):
                continue

            if considered_regexps:
                if any([r.match(dirpath) for r in considered_regexps]):
                    continue

            if drs == "None":
                raise NotImplementedError("Raise an issue if you need this")

            scmcube = _CUBES[drs]()
            ids = {
                k: v
                if any(
                    [s in k for s in ["variable", "experiment", "activity", "mip"]]
                )
                else ".*"
                for k, v in scmcube.process_path(dirpath).items()
            }

            regexp_here = re.compile(
                dirname(
                    scmcube.get_filepath_from_load_data_from_identifiers_args(**ids)
                )
            )
            logger.info("Wrangling {}".format(regexp_here))
            regexp_compiled = re.compile(regexp)

            format_custom_text = _get_format_custom_text()
            bar = _get_progressbar(
                text=format_custom_text, max_value=len([w for w in walk(src)])
            )
            collected = []
            for i, (dirpath, dirnames, filenames) in enumerate(walk(src)):
                if filenames:
                    if not regexp_compiled.match(dirpath):
                        continue

                    format_custom_text.update_mapping(curr_dir=dirpath)
                    bar.update(i)

                    openscmdf = df_append([join(dirpath, f) for f in filenames])
                    tmp_ts = openscmdf.timeseries().reset_index()
                    tmp_ts["unit"] = tmp_ts["unit"].astype(str)
                    openscmdf = ScmDataFrame(tmp_ts)

                    collected.append(openscmdf)

            _tuningstrucs_wrangling(dst, regexp, df_append(collected), force, prefix, blend_models=True)

            considered_regexps.append(regexp_here)


def _do_wrangling(src, dst, regexp, nested, out_format, force, prefix):
    regexp_compiled = re.compile(regexp)

    format_custom_text = _get_format_custom_text()
    bar = _get_progressbar(
        text=format_custom_text, max_value=len([w for w in walk(src)])
    )
    for i, (dirpath, dirnames, filenames) in enumerate(walk(src)):
        if filenames:
            if not regexp_compiled.match(dirpath):
                continue

            format_custom_text.update_mapping(curr_dir=dirpath)
            bar.update(i)

            openscmdf = df_append([join(dirpath, f) for f in filenames])
            tmp_ts = openscmdf.timeseries().reset_index()
            tmp_ts["unit"] = tmp_ts["unit"].astype(str)
            openscmdf = ScmDataFrame(tmp_ts)

            out_filedir = dirpath.replace(src, dst) if nested else dst
            _make_path_if_not_exists(out_filedir)

            if out_format == "tuningstrucs":
                _tuningstrucs_wrangling(out_filedir, filenames, openscmdf, force, prefix)
            else:
                raise ValueError("Unsupported format: {}".format(out_format))


def _tuningstrucs_wrangling(out_root_dir, source_info, source_openscmdf, force, prefix, blend_models=False):
    sdf_iter = [source_openscmdf] if blend_models else [source_openscmdf.filter(climate_model=m) for m in source_openscmdf["climate_model"]]
    for sdf in sdf_iter:
        logger.info("Wrangling {}".format(source_info))
        convert_scmdf_to_tuningstruc(
            sdf, out_root_dir, force=force, prefix=prefix,
        )


def _get_timestamp():
    return strftime("%Y%m%d %H%M%S", gmtime())


def _make_path_if_not_exists(path_to_check):
    if not path.exists(path_to_check):
        logger.info("Making output directory: {}".format(path_to_check))
        makedirs(path_to_check)


def _get_format_custom_text():
    return progressbar.FormatCustomText(
        "Current directory :: %(curr_dir)-400s", {"curr_dir": "uninitialised"}
    )


def _get_progressbar(text, max_value):
    return progressbar.ProgressBar(
        widgets=[progressbar.SimpleProgress(), ". ", text],
        max_value=max_value,
        prefix="Visiting directory ",
    ).start()
