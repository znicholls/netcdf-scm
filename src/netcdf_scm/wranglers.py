from os.path import isfile


import numpy as np
from openscm.scmdataframe import ScmDataFrame


from . import mat4py


def convert_tuningstruc_to_scmdf(
    filepath, variable=None, region=None, unit=None, scenario=None, model=None
):
    """Convert a matlab tuningstruc to an ScmDataFrame

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
    ValueError
        If a metadata variable is not supplied and it cannot be determined from the
        tuningstruc.

    Returns
    -------
    :obj: `ScmDataFrame`
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

        scmdf = ScmDataFrame(
            data=np.asarray(dataset["tuningdata"]["model"][m]["data"][1]),
            index=dataset["tuningdata"]["model"][m]["data"][0],
            columns=metadata,
        )

        try:
            ref_df.append(scmdf, inplace=True)
        except NameError:
            ref_df = scmdf
    return ref_df


def convert_scmdf_to_tuningstruc(scmdf, outpath, force=False):
    """Convert an ScmDataFrame to a matlab tuningstruc

    One tuningstruc file will be created for each unique
    ["model", "scenario", "variable", "region", "unit"] combination in the input
    ScmDataFrame.

    Parameters
    ----------
    scmdf : :obj: `ScmDataFrame`
        ScmDataFrame to convert to a tuningstruc

    outpath : str
        Base path in which to save the tuningstruc. The rest of the pathname is
        generated from the metadata. `.mat` is also appended automatically.

    force : bool
        If True, overwrite any existing files

    Returns
    -------
    list
        List of files which were not re-written as they already exist
    """
    already_written = []

    iterable = scmdf.timeseries().groupby(
        ["model", "scenario", "variable", "region", "unit"]
    )
    for (model, scenario, variable, region, unit), df in iterable:
        dataset = {}
        dataset["tuningdata"] = {}
        dataset["tuningdata"]["modelcodes"] = []
        dataset["tuningdata"]["model"] = []

        for m, (climate_model, cmdf) in enumerate(df.groupby("climate_model")):
            # impossible to make dataframe with duplicate rows, this is just in
            # case
            error_msg = (
                "Should only have a single unique timeseries for a given "
                '["climate_model", "model", "scenario", "variable", '
                '"region", "unit"] combination'
            )
            assert cmdf.shape[0] == 1, error_msg

            dataset["tuningdata"]["modelcodes"].append(climate_model)
            dataset["tuningdata"]["model"].append({})

            dataset["tuningdata"]["model"][m]["model"] = model
            dataset["tuningdata"]["model"][m]["scenario"] = scenario
            dataset["tuningdata"]["model"][m]["variable"] = variable
            dataset["tuningdata"]["model"][m]["region"] = region
            dataset["tuningdata"]["model"][m]["unit"] = unit

            dataset["tuningdata"]["model"][m]["notes"] = (
                "{} {} {} {} ({}) tuningstruc (written with scmcallib)"
                "".format(scenario, model, region, variable, unit)
            )
            dataset["tuningdata"]["model"][m]["data"] = [
                [float(t.year) for t in cmdf.columns],
                list(cmdf.values.squeeze()),
            ]
            dataset["tuningdata"]["model"][m]["col_code"] = ["YEARS", variable]

        outfile = get_tuningstruc_name_from_df(df, outpath)

        if isfile(outfile) and not force:
            already_written.append(outfile)
        else:
            mat4py.savemat(outfile, dataset)

    return already_written


def get_tuningstruc_name_from_df(df, outpath):
    """
    Get the name of a tuningstruc from a ``pd.DataFrame``

    Parameters
    ----------
    df : :obj: `pd.DataFrame`
        *pandas* DataFrame to convert to a tuningstruc

    outpath : str
        Base path on which to append the metadata and `.mat`.

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

    def _get_col(col):
        try:
            vals = df[col].unique()
        except KeyError:
            vals = df.index.get_level_values(col).unique()
        if len(vals) != 1:
            raise ValueError("More than one {} in df".format(col))

        return vals[0]

    scenario = _get_col("scenario")
    model = _get_col("model")
    variable = _get_col("variable")
    region = _get_col("region")

    return (
        "{}_{}_{}_{}_{}.mat".format(outpath, scenario, model, variable, region)
        .replace(" ", "_")
        .replace("|", "_")
    )
