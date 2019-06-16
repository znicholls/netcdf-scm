try:
    import iris
except ModuleNotFoundError:  # pragma: no cover # emergency valve
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()

from .iris_cube_wrappers import SCMCube
from .definitions import _SCM_TIMESERIES_META_COLUMNS

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
    scmdf = helper._convert_scm_timeseries_cubes_to_openscmdata(scm_cubes)
    scmdf.metadata = {
        k:v for k, v in helper.cube.attributes.items() if k != "region"
    }
    for coord in helper.cube.coords():
        if coord.standard_name in ["time", "latitude", "longitude", "height"]:
            continue
        elif coord.long_name.startswith("land_fraction"):
            scmdf.metadata[coord.long_name] = coord.points.squeeze()
        else:
            # this is really how it should work for land_fraction too but we don't
            # have a stable solution for parameter handling in OpenSCMDataFrame yet so
            # I've done the above instead
            extra_str = "{} ({})".format(coord.long_name, str(coord.units))
            scmdf[extra_str] = coord.points.squeeze()

    return scmdf


def _load_helper_and_scm_cubes(path):
    cube_list = iris.load(path)

    loaded = SCMCube()
    scm_cubes = {}
    for i, v in enumerate(cube_list):
        region = v.attributes["region"]
        scm_cubes[region] = SCMCube()
        scm_cubes[region].cube = v
        # take any cube as base for now, not sure how to really handle this so will
        # leave like this for now and only make this method public when I work it
        # out...
        if i == 0:
            loaded.cube = v

    return loaded, scm_cubes
