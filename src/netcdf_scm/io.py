"""Input and output from NetCDF-SCM's netCDF format"""
try:
    import iris
except ModuleNotFoundError:  # pragma: no cover # emergency valve
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()

from .definitions import _SCM_TIMESERIES_META_COLUMNS
from .iris_cube_wrappers import SCMCube


def save_netcdf_scm_nc(cubes, out_path):
    """
    Save a series of cubes to a `.nc` file

    Parameters
    ----------
    cubes : dict
        Dictionary of "region name"-:obj:`ScmCube` key-value pairs. The cubes will all
        be saved in the same ``.nc`` file.

    out_path : str
        Path in which to save the data
    """
    save_cubes = []
    for scm_cube in cubes.values():
        cube = scm_cube.cube
        save_cubes.append(cube)

    iris.save(save_cubes, out_path, local_keys=_SCM_TIMESERIES_META_COLUMNS)


def load_scmdataframe(path):
    """
    Load an scmdataframe from a NetCDF-SCM ``.nc`` file

    Parameters
    ----------
    path : str
        Path from which to load the data

    Returns
    -------
    :obj:`ScmDataFrame`
        :obj:`ScmDataFrame` containing the data in ``path``.
    """
    helper, scm_cubes = _load_helper_and_scm_cubes(path)
    scmdf = helper.convert_scm_timeseries_cubes_to_openscmdata(scm_cubes)

    # TODO: decide whether assuming point data is a good idea or not
    return scmdf


def _load_helper_and_scm_cubes(path):
    cube_list = iris.load(path)

    loaded = SCMCube()
    scm_cubes = {}
    for v in cube_list:
        region = v.attributes["region"]
        scm_cubes[region] = SCMCube()
        scm_cubes[region].cube = v

    # take any cube as base for now, not sure how to really handle this so will
    # leave like this for now and only make this method public when I work it
    # out...
    loaded.cube = cube_list[0]

    return loaded, scm_cubes
