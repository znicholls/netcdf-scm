"""Miscellaneous readers for files which can't otherwise be read"""
from scmdata import ScmDataFrame

from .iris_cube_wrappers import SCMCube
from .utils import _check_cube_and_adjust_if_needed

try:
    import iris

except ModuleNotFoundError:  # pragma: no cover # emergency valve
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()


def read_cmip6_concs_gmnhsh(  # pylint:disable=too-many-locals
    filepath, region_coord_name="sector"
):
    """
    Read CMIP6 concentrations global and hemispheric mean data

    Parameters
    ----------
    filepath : str
        Filepath from which to read the data

    region_coord_name : str
        The name of the co-ordinate which represents the region in the datafile.

    Returns
    -------
    :obj:`ScmDataFrame`
        :obj:`ScmDataFrame` containing the global and hemispheric mean data

    Raises
    ------
    AssertionError
        Defensive assertion: the code is being used in an unexpected way
    """
    loaded_cube = iris.load_cube(filepath)
    checked_cube = _check_cube_and_adjust_if_needed(loaded_cube)

    region_map = {
        "GM": "World",
        "NH": "World|Northern Hemisphere",
        "SH": "World|Southern Hemisphere",
    }
    unit_map = {"1.e^-6": "ppm", "1.e^-9": "ppb", "1.e^-12": "ppt"}

    timeseries_cubes = {}
    for region_coord in checked_cube.coord(region_coord_name):
        if len(list(region_coord.cells())) != 1:  # pragma: no cover
            raise AssertionError("Should only have one point now")

        original_names = {
            int(v.split(":")[0].strip()): v.split(":")[1].strip()
            for v in region_coord.attributes["original_names"].split(";")
        }
        original_regions = {k: v.split("_")[-1] for k, v in original_names.items()}
        region_coord_point = region_coord.cell(0).point
        region = region_map[original_regions[region_coord_point]]
        if checked_cube.shape[1] != 3 or checked_cube.shape[0] == 3:
            raise AssertionError("cube data shape isn't as expected")

        checked_cube.attributes["variable"] = checked_cube.var_name
        checked_cube.attributes["variable_standard_name"] = checked_cube.standard_name
        checked_cube.attributes["region"] = region
        if checked_cube.attributes["source_id"].startswith("UoM-CMIP"):
            scenario = "historical"
            model = "unspecified"
        else:
            scenario = "-".join(
                "ssp{}".format(
                    checked_cube.attributes["source_id"].split("ssp")[1]
                ).split("-")[:-3]
            )
            model = (
                checked_cube.attributes["source_id"]
                .split("-ssp")[0]
                .replace("UoM-", "")
            )
        checked_cube.attributes["scenario"] = scenario
        checked_cube.attributes["model"] = model
        checked_cube.attributes["climate_model"] = "MAGICC7"
        checked_cube.attributes["member_id"] = "unspecified"

        helper_region = SCMCube()
        helper_region.cube = checked_cube[:, region_coord_point]
        helper_region.cube.remove_coord(region_coord_name)
        timeseries_cubes[region] = helper_region

    output = (
        helper_region.convert_scm_timeseries_cubes_to_openscmdata(timeseries_cubes)
        .timeseries()
        .reset_index()
    )
    output["unit"] = output["unit"].map(unit_map)
    output["model"] = model
    output = ScmDataFrame(output)

    return output
