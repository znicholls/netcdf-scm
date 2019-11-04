"""Functions used to 'wrangle' netCDF-SCM netCDF files into other formats"""
import logging
from os.path import isfile, join

import numpy as np
from scmdata import ScmDataFrame

from . import mat4py

logger = logging.getLogger(__name__)


def convert_tuningstruc_to_scmdf(  # pylint:disable=too-many-arguments,too-many-locals
    filepath, variable=None, region=None, unit=None, scenario=None, model=None
):
    """
    Convert a matlab tuningstruc to an ScmDataFrame

    Parameters
    ----------
    filepath : str
        Filepath from which to load the data

    variable : str
        Name of the variable contained in the tuningstruc. If None,
        `convert_tuningstruc_to_scmdf` will attempt to determine it from the input
        file.

    region : str
        Region to which the data in the tuningstruc applies. If None,
        `convert_tuningstruc_to_scmdf` will attempt to determine it from the input
        file.

    unit : str
        Units of the data in the tuningstruc. If None,
        `convert_tuningstruc_to_scmdf` will attempt to determine it from the input
        file.

    scenario : str
        Scenario to which the data in the tuningstruc applies. If None,
        `convert_tuningstruc_to_scmdf` will attempt to determine it from the input
        file.

    model : str
        The (integrated assessment) model which generated the emissions scenario
        associated with the data in the tuningstruc. If None,
        `convert_tuningstruc_to_scmdf` will attempt to determine it from the input
        file and if it cannot, it will be set to "unspecified".

    Raises
    ------
    KeyError
        If a metadata variable is not supplied and it cannot be determined from the
        tuningstruc.

    Returns
    -------
    :obj:`ScmDataFrame`
        ScmDataFrame with the tuningstruc data
    """
    dataset = mat4py.loadmat(filepath)

    for m, climate_model in enumerate(dataset["tuningdata"]["modelcodes"]):
        metadata = {
            "variable": [variable],
            "region": [region],
            "unit": [unit],
            "climate_model": [climate_model],
            "scenario": [scenario],
            "model": [model],
        }
        for k, v in metadata.items():
            if v == [None]:
                try:
                    metadata[k] = [dataset["tuningdata"]["model"][m][k]]
                except KeyError:
                    if k == "model":
                        metadata[k] = ["unspecified"]
                        continue

                    error_msg = "Cannot determine {} from file: " "{}".format(
                        k, filepath
                    )
                    raise KeyError(error_msg)

        data = np.asarray(dataset["tuningdata"]["model"][m]["data"])
        if len(data) != 2:
            data = data.T
        scmdf = ScmDataFrame(data=data[1], index=data[0], columns=metadata)

        try:
            ref_df.append(scmdf, inplace=True)
        except NameError:
            ref_df = scmdf
    return ref_df


def convert_scmdf_to_tuningstruc(scmdf, outdir, prefix=None, force=False):
    """
    Convert an ScmDataFrame to a matlab tuningstruc

    One tuningstruc file will be created for each unique
    ["model", "scenario", "variable", "region", "unit"] combination in the input
    ScmDataFrame.

    Parameters
    ----------
    scmdf : :obj:`ScmDataFrame`
        ScmDataFrame to convert to a tuningstruc

    outdir : str
        Directory in which to save the tuningstruc

    prefix : str
        Prefix for the filename. The rest of the filename is generated from the
        metadata. `.mat` is also appended automatically. If ``None``, no prefix
        is used.

    force : bool
        If True, overwrite any existing files

    Returns
    -------
    list
        List of files which were not re-written as they already exist

    Raises
    ------
    AssertionError
        If timeseries are not unique for a given ["climate_model", "model",
        "scenario", "variable", "region", "unit"] combination.
    """
    already_written = []

    group_cols = ["model", "scenario", "variable", "region", "unit"]
    for label, df in scmdf.timeseries().groupby(group_cols):
        ids = dict(zip(group_cols, label))

        dataset = {}
        dataset["tuningdata"] = {}
        dataset["tuningdata"]["modelcodes"] = []
        dataset["tuningdata"]["model"] = []

        for m, (climate_model, cmdf) in enumerate(df.groupby("climate_model")):
            # impossible to make dataframe with duplicate rows, this is just in
            # case
            if cmdf.shape[0] != 1:  # pragma: no cover # emergency valve
                raise AssertionError(
                    "Should only have a single unique timeseries for a given "
                    '["climate_model", "model", "scenario", "variable", '
                    '"region", "unit"] combination'
                )

            dataset["tuningdata"]["modelcodes"].append(climate_model)
            dataset["tuningdata"]["model"].append({})

            dataset["tuningdata"]["model"][m]["model"] = ids["model"]
            dataset["tuningdata"]["model"][m]["scenario"] = ids["scenario"]
            dataset["tuningdata"]["model"][m]["variable"] = ids["variable"]
            dataset["tuningdata"]["model"][m]["region"] = ids["region"]
            dataset["tuningdata"]["model"][m]["unit"] = ids["unit"]

            dataset["tuningdata"]["model"][m]["notes"] = (
                "{} {} {} {} ({}) tuningstruc (written with scmcallib)"
                "".format(
                    ids["scenario"],
                    ids["model"],
                    ids["region"],
                    ids["variable"],
                    ids["unit"],
                )
            )
            dataset["tuningdata"]["model"][m]["data"] = [
                list(e)
                for e in zip(
                    [float(t.year) for t in cmdf.columns],
                    list(cmdf.values.squeeze()),
                )
            ]
            dataset["tuningdata"]["model"][m]["col_code"] = ["YEARS", ids["variable"]]

        outfile = get_tuningstruc_name_from_df(df, outdir, prefix)

        if isfile(outfile) and not force:
            logger.info("Skipped (already exists, not overwriting) %s", outfile)
            already_written.append(outfile)
        else:
            logger.info("Writing %s", outfile)
            mat4py.savemat(outfile, dataset)

    return already_written


def get_tuningstruc_name_from_df(df, outdir, prefix):
    """
    Get the name of a tuningstruc from a ``pd.DataFrame``

    Parameters
    ----------
    df : :obj:`pd.DataFrame`
        *pandas* DataFrame to convert to a tuningstruc

    outdir : str
        Base path on which to append the metadata and `.mat`.

    prefix : str
        Prefix to prepend to the name. If ``None``, no prefix is prepended.

    Returns
    -------
    str
        tuningstruc name

    Raises
    ------
    ValueError
        A name cannot be determined because e.g. more than one scenario is contained
        in the dataframe
    """
    prefix = "{}_".format(prefix) if prefix is not None else ""

    def _get_col(col):
        try:
            vals = df[col].unique()
        except KeyError:
            vals = df.index.get_level_values(col).unique()
        if len(vals) != 1:
            raise ValueError("More than one {} in df".format(col))

        return vals[0]

    scenario = _get_col("scenario")
    try:
        member_id = _get_col("member_id")
    except KeyError:
        member_id = "member-id"
    variable = _get_col("variable")
    region = _get_col("region")
    raw_name = (
        (
            "{}_{}_{}_{}".format(variable, scenario, member_id, region)
            .replace(" ", "_")
            .replace("|", "_")
        )
    ).upper() + ".mat"

    return join(outdir, "{}{}".format(prefix, raw_name))
