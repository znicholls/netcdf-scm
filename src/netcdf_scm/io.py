try:
    import iris
except ModuleNotFoundError:  # pragma: no cover # emergency valve
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()

from netcdf_scm.iris_cube_wrappers import SCMCube


_IO_LOCAL_KEYS = ["region"]

def save_netcdf_scm_nc(cubes, out_path):
    """
    Save a series of cubes to a `.nc` file

    Parameters
    ----------
    cube : dict
        Dictionary of "region name"-:obj:`ScmCube` key-value pairs. The cubes will all be saved in the same ``.nc`` file.

    out_path : str
        Path in which to save the data
    """
    save_cubes = []
    for region, scm_cube in cubes.items():
        cube = scm_cube.cube
        cube.attributes["region"] = region
        save_cubes.append(cube)

    iris.save(save_cubes, out_path, local_keys=_IO_LOCAL_KEYS)


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
    helper = SCMCube()
    cube_list = iris.load(path)
    scm_cubes = {}
    for v in cube_list:
        region = v.attributes["region"]
        scm_cubes[region] = SCMCube()
        scm_cubes[region].cube = v

    helper.cube = v  # need any cube to get e.g. variable attributes

    return helper._convert_scm_timeseries_cubes_to_openscmdata(scm_cubes)


def load_netcdf_scm_nc(path):
    """
    Load data from a NetCDF-SCM ``.nc`` file

    Parameters
    ----------
    path : str
        Path from which to load the data

    Returns
    -------
    dict
        Dictionary of "region name"-:obj:`ScmCube` key-value pairs.
    """
    import pdb
    pdb.set_trace()
