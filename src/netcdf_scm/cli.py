"""Command line interface"""
import datetime as dt
import logging
import os
import os.path
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from os import makedirs, walk
from time import gmtime, strftime

import click
import numpy as np
import pandas as pd
import pymagicc
import tqdm
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

try:
    import dask
except ModuleNotFoundError:  # pragma: no cover # emergency valve
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()

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
    default="^(?!.*(fx)).*$",
    show_default=True,
    help="Regular expression to apply to file directory (only crunches matches).",
)
@click.option(
    "--regions",
    default="World,World|Northern Hemisphere,World|Southern Hemisphere,World|Land,World|Ocean,World|Northern Hemisphere|Land,World|Southern Hemisphere|Land,World|Northern Hemisphere|Ocean,World|Southern Hemisphere|Ocean",
    show_default=True,
    help="Comma-separated regions to crunch.",
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
@click.option(
    "--small-number-workers",
    default=10,
    show_default=True,
    help="Maximum number of workers to use when crunching files.",
)
@click.option(
    "--small-threshold",
    default=50.0,
    show_default=True,
    help="Maximum number of data points (in millions) in a file for it to be processed in parallel with ``small-number-workers``",
)
@click.option(
    "--medium-number-workers",
    default=3,
    show_default=True,
    help="Maximum number of workers to use when crunching files.",
)
@click.option(
    "--medium-threshold",  # pylint:disable=too-many-arguments,too-many-locals,too-many-statements
    default=120.0,
    show_default=True,
    help="Maximum number of data points (in millions) in a file for it to be processed in parallel with ``medium-number-workers``",
)
@click.option(
    "--force-lazy-threshold",
    default=1000.0,
    show_default=True,
    help="Maximum number of data points (in millions) in a file for it to be processed in memory",
)
def crunch_data(
    src,
    dst,
    crunch_contact,
    drs,
    regexp,
    regions,
    data_sub_dir,
    force,
    small_number_workers,
    small_threshold,
    medium_number_workers,
    medium_threshold,
    force_lazy_threshold,
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
            ("regions", regions),
            ("force", force),
            ("small_number_workers", small_number_workers),
            ("small_threshold", small_threshold),
            ("medium_number_workers", medium_number_workers),
            ("medium_threshold", medium_threshold),
            ("force_lazy_threshold", force_lazy_threshold),
        ],
        out_filename=log_file,
    )

    tracker = OutputFileDatabase(out_dir)
    regexp_to_match = re.compile(regexp)
    helper = _get_scmcube_helper(drs)

    def keep_dir(dpath):
        if not regexp_to_match.match(dpath):
            logger.debug("Skipping (did not match regexp) %s", dpath)
            return False
        logger.info("Adding directory to queue %s", dpath)

        return True

    found_dirs, failures_dir_finding = _find_dirs_meeting_func(src, keep_dir)

    def get_number_data_points_in_millions(dpath_h):
        try:
            helper.load_data_in_directory(dpath_h, process_warnings=False)
        except Exception as e:  # pylint:disable=broad-except
            logger.exception(
                "Could not calculate size of data in %s, exception: %s", dpath_h, e
            )
            return None

        data_points = np.prod(helper.cube.shape) / 10 ** 6
        logger.debug("data in %s has %s million data points", dpath_h, data_points)
        return data_points

    failures_calculating_data_points = False
    dirs_to_crunch = []
    for d, f in tqdm.tqdm(found_dirs, desc="Sorting directories"):
        p = get_number_data_points_in_millions(d)
        if p is None:
            failures_calculating_data_points = True
        else:
            dirs_to_crunch.append((d, f, p))

    crunch_kwargs = {
        "drs": drs,
        "separator": separator,
        "output_prefix": output_prefix,
        "out_dir": out_dir,
        "regions": regions,
        "force": force,
        "existing_files": tracker._data,  # pylint:disable=protected-access
        "crunch_contact": crunch_contact,
        "force_lazy_threshold": force_lazy_threshold,
    }

    def process_results(res):
        if res is None:
            return  # skipped crunching
        scm_timeseries_cubes, out_filepath, info = res
        logger.info("Registering %s", out_filepath)
        tracker.register(out_filepath, info)
        logger.info("Writing file to %s", out_filepath)
        save_netcdf_scm_nc(scm_timeseries_cubes, out_filepath)

    def crunch_from_list(crunch_list, n_workers=1):
        return _apply_func(
            _crunch_files,
            crunch_list,
            common_kwarglist=crunch_kwargs,
            postprocess_func=process_results,
            n_workers=n_workers,
            style="processes",
        )

    failures_small = False
    dirs_to_crunch_small = [
        {"fnames": f, "dpath": d} for d, f, n in dirs_to_crunch if n < small_threshold
    ]
    logger.info(
        "Crunching %s directories with less than %s million data points",
        len(dirs_to_crunch_small),
        small_threshold,
    )
    if dirs_to_crunch_small:
        failures_small = crunch_from_list(
            dirs_to_crunch_small, n_workers=small_number_workers
        )

    failures_medium = False
    dirs_to_crunch_medium = [
        {"fnames": f, "dpath": d}
        for d, f, n in dirs_to_crunch
        if small_threshold <= n < medium_threshold
    ]
    logger.info(
        "Crunching %s directories with greater than or equal to %s and less than %s million data points",
        len(dirs_to_crunch_medium),
        small_threshold,
        medium_threshold,
    )
    if dirs_to_crunch_medium:
        failures_medium = crunch_from_list(
            dirs_to_crunch_medium, n_workers=medium_number_workers
        )

    failures_large = False
    dirs_to_crunch_large = [
        {"fnames": f, "dpath": d} for d, f, n in dirs_to_crunch if n > medium_threshold
    ]
    logger.info(
        "Crunching %s directories with greater than or equal to %s million data points",
        len(dirs_to_crunch_large),
        medium_threshold,
    )
    if dirs_to_crunch_large:
        failures_large = crunch_from_list(dirs_to_crunch_large, n_workers=1)

    if (
        failures_calculating_data_points
        or failures_dir_finding
        or failures_small
        or failures_medium
        or failures_large
    ):
        raise click.ClickException(
            "Some files failed to process. See {} for more details".format(log_file)
        )


def _crunch_files(  # pylint:disable=too-many-arguments,too-many-locals
    fnames,
    dpath,
    drs=None,
    separator=None,
    output_prefix=None,
    out_dir=None,
    regions=None,
    force=None,
    existing_files=None,
    crunch_contact=None,
    force_lazy_threshold=None,
):
    logger.info("Attempting to process: %s", fnames)
    scmcube = _load_scm_cube(drs, dpath, fnames)

    out_filename = separator.join([output_prefix, scmcube.get_data_filename()])

    outfile_dir = scmcube.get_data_directory().replace(scmcube.root_dir, out_dir)
    out_filepath = os.path.join(outfile_dir, out_filename)

    _make_path_if_not_exists(outfile_dir)

    if not force and out_filepath in existing_files:
        logger.info("Skipped (already exists, not overwriting) %s", out_filepath)
        return None

    regions = regions.split(",")
    if scmcube.netcdf_scm_realm == "ocean":
        ocean_regions = [r for r in regions if "Land" not in r]
        if set(regions) - set(ocean_regions):
            regions = ocean_regions
            logger.warning(
                "Detected ocean data, dropping land related regions so regions "
                "to crunch are now: %s",
                regions,
            )

    elif scmcube.netcdf_scm_realm == "land":
        land_regions = [
            r for r in regions if not any([ss in r for ss in ("Ocean", "El Nino")])
        ]
        if set(regions) - set(land_regions):
            regions = land_regions
            logger.warning(
                "Detected land data, dropping ocean related regions so regions "
                "to crunch are now: %s",
                regions,
            )

    ndata_points = np.prod(scmcube.cube.shape) / 10 ** 6
    lazy = ndata_points > force_lazy_threshold
    if lazy:
        logger.info(
            "Data in %s has %s million data points which is above "
            "force-lazy-threshold of %s million data points hence processing lazily",
            dpath,
            ndata_points,
            force_lazy_threshold,
        )
    results = scmcube.get_scm_timeseries_cubes(regions=regions, lazy=lazy)
    results = _set_crunch_contact_in_results(results, crunch_contact)

    return results, out_filepath, scmcube.info


def _find_dirs_meeting_func(src, check_func):
    matching_dirs = []
    failures = False
    logger.info("Finding directories with files")
    for dirpath, _, filenames in walk(src):
        logger.debug("Entering %s", dirpath)
        if filenames:
            try:
                if check_func(dirpath):
                    matching_dirs.append((dirpath, filenames))
            except Exception as e:  # pylint:disable=broad-except
                logger.error(
                    "Directory checking failed on %s with error %s", dirpath, e
                )
                failures = True

    logger.info("Found %s directories with files", len(matching_dirs))
    return matching_dirs, failures


def _apply_func(  # pylint:disable=too-many-arguments
    apply_func,
    loop_kwarglist,
    common_arglist=None,
    common_kwarglist=None,
    postprocess_func=None,
    n_workers=2,
    style="processes",
):
    common_arglist = [] if common_arglist is None else common_arglist
    common_kwarglist = {} if common_kwarglist is None else common_kwarglist
    tqdm_kwargs = {
        "total": len(loop_kwarglist),
        "unit": "it",
        "unit_scale": True,
        "leave": True,
    }
    if n_workers == 1:
        failures = _apply_func_serially(
            apply_func=apply_func,
            loop_kwarglist=loop_kwarglist,
            tqdm_kwargs=tqdm_kwargs,
            common_arglist=common_arglist,
            common_kwarglist=common_kwarglist,
            postprocess_func=postprocess_func,
        )
    else:
        failures = _apply_func_parallel(
            apply_func=apply_func,
            loop_kwarglist=loop_kwarglist,
            tqdm_kwargs=tqdm_kwargs,
            common_arglist=common_arglist,
            common_kwarglist=common_kwarglist,
            postprocess_func=postprocess_func,
            n_workers=n_workers,
            style=style,
        )

    return failures


def _apply_func_serially(  # pylint:disable=too-many-arguments
    apply_func,
    loop_kwarglist,
    tqdm_kwargs,
    common_arglist,
    common_kwarglist,
    postprocess_func,
):
    failures = False
    logger.info("Processing serially")
    for ikwargs in tqdm.tqdm(loop_kwarglist, **tqdm_kwargs):
        try:
            res = apply_func(*common_arglist, **ikwargs, **common_kwarglist)
            if postprocess_func is not None:
                postprocess_func(res)
        except Exception as e:  # pylint:disable=broad-except
            logger.exception("Exception found %s", e)
            failures = True

    return failures


def _apply_func_parallel(  # pylint:disable=too-many-arguments
    apply_func,
    loop_kwarglist,
    tqdm_kwargs,
    common_arglist,
    common_kwarglist,
    postprocess_func,
    n_workers,
    style,
):
    failures = False
    logger.info("Processing in parallel with %s workers", n_workers)
    logger.info("Forcing dask to use a single thread when reading")
    with dask.config.set(scheduler="single-threaded"):
        if style == "processes":
            executor_cls = ProcessPoolExecutor
        elif style == "threads":
            executor_cls = ThreadPoolExecutor
        else:
            raise ValueError("Unrecognised executor: {}".format(style))
        with executor_cls(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    apply_func, *common_arglist, **ikwargs, **common_kwarglist
                )
                for ikwargs in loop_kwarglist
            ]
            failures = False
            # Print out the progress as tasks complete
            for future in tqdm.tqdm(as_completed(futures), **tqdm_kwargs):
                try:
                    res = future.result()
                    if postprocess_func is not None:
                        postprocess_func(res)
                except Exception as e:  # pylint:disable=broad-except
                    logger.exception("Exception found %s", e)
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
    default="^(?!.*(fx)).*$",
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
    "--force/--do-not-force",
    "-f",
    help="Overwrite any existing files.",
    default=False,
    show_default=True,
)  # pylint:disable=too-many-arguments
@click.option(
    "--number-workers",  # pylint:disable=too-many-arguments
    help="Number of worker (threads) to use when wrangling.",
    default=4,
    show_default=True,
)
@click.option(
    "--target-units-specs",  # pylint:disable=too-many-arguments
    help="csv containing target units for wrangled variables.",
    default=None,
    show_default=False,
    type=click.Path(exists=True, readable=True, resolve_path=True)
)
def wrangle_netcdf_scm_ncs(
    src, dst, wrangle_contact, regexp, prefix, out_format, drs, force, number_workers, target_units_specs
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
        _do_wrangling(
            src, dst, regexp, out_format, force, wrangle_contact, drs, number_workers, target_units_specs
        )


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
        if any(
            [s in k for s in ["variable", "experiment", "activity", "mip", "member_id"]]
        )
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
    src, dst, regexp, out_format, force, wrangle_contact, drs, number_workers, target_units_specs
):
    regexp_compiled = re.compile(regexp)
    if target_units_specs is not None:
        target_units_specs = pd.read_csv(target_units_specs)

    if out_format in ("mag-files", "magicc-input-files-point-end-of-year"):
        _do_magicc_wrangling(
            src,
            dst,
            regexp_compiled,
            out_format,
            force,
            wrangle_contact,
            drs,
            number_workers,
            target_units_specs,
        )
    else:  # pragma: no cover # emergency valve (should be caught by click on call)
        raise ValueError("Unsupported format: {}".format(out_format))


def _do_magicc_wrangling(  # pylint:disable=too-many-arguments,too-many-locals
    src, dst, regexp_compiled, out_format, force, wrangle_contact, drs, number_workers, target_units_specs
):
    scmcube = _get_scmcube_helper(drs)
    crunch_list, failures_dir_finding = _find_dirs_meeting_func(
        src, regexp_compiled.match
    )

    def get_openscmdf_metadata_header(fnames, dpath):
        openscmdf = df_append(
            [load_scmdataframe(os.path.join(dpath, f)) for f in fnames]
        )
        if target_units_specs is not None:
            for variable in openscmdf["variable"].unique():
                if variable in target_units_specs["variable"].tolist():
                    target_unit = target_units_specs[
                        target_units_specs["variable"] == variable
                    ]["unit"].values[0]
                    openscmdf = openscmdf.convert_unit(target_unit, variable=variable)

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
        failures_wrangling = _apply_func(
            wrangle_to_mag_files,
            [{"fnames": f, "dpath": d} for d, f in crunch_list],
            n_workers=number_workers,
            style="threads",
        )

    else:  # out_format == "magicc-input-files-point-end-of-year":
        wrangle_to_magicc_input_files_point_end_of_year = _get_wrangle_to_magicc_input_files_point_end_of_year_func(
            force, get_openscmdf_metadata_header, get_outfile_dir_symlink_dir
        )
        failures_wrangling = _apply_func(
            wrangle_to_magicc_input_files_point_end_of_year,
            [{"fnames": f, "dpath": d} for d, f in crunch_list],
            n_workers=number_workers,
            style="threads",
        )

    if failures_dir_finding or failures_wrangling:
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
        symlink_file = os.path.join(symlink_dir, os.path.basename(out_file))
        os.unlink(symlink_file)
        os.remove(out_file)

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
