"""
Module which handles masking of data

The masks are applied to data fields to exclude unwanted parts of the globe. The
convention used here is the same as numpy masked arrays, where ``True`` values are
excluded.
"""
import logging
import os

import numpy as np

from ..utils import broadcast_onto_lat_lon_grid

try:
    import iris
except ModuleNotFoundError:  # pragma: no cover # emergency valve
    from ..errors import raise_no_iris_warning

    raise_no_iris_warning()


logger = logging.getLogger(__name__)

_DEFAULT_SFTLF_FILE = "default_land_ocean_mask.nc"
DEFAULT_REGIONS = (
    "World",
    "World|Land",
    "World|Ocean",
    "World|Northern Hemisphere",
    "World|Southern Hemisphere",
    "World|Northern Hemisphere|Land",
    "World|Southern Hemisphere|Land",
    "World|Northern Hemisphere|Ocean",
    "World|Southern Hemisphere|Ocean",
)


class InvalidMask(Exception):
    """
    Raised when a mask cannot be calculated.

    This error usually propogates. For example, if a child mask used in the
    calculation of a parent mask fails then the parent mask should also raise an
    InvalidMask exception (unless it can be satisfactorily handled).
    """


def invert(mask_to_invert):
    """
    Invert a mask

    e.g. convert from Land to Ocean

    Parameters
    ----------
    mask_to_invert: str
        Name of the mask to invert. This mask is loaded at evaluation time.

    Returns
    -------
    MaskFunc
    """

    def f(masker, cube, **kwargs):
        return ~masker.get_mask(mask_to_invert)

    return f


def or_masks(mask_a, mask_b):
    """
    Take the 'or' product of two masks

    This is the equivilent of an inner join between two masks. Only values which are
    not masked in both masks (False in both masks) will remain unmasked (False).

    Parameters
    ----------
    mask_a : str or MaskFunc
        If a string is provided, the mask specified by the string is retrieved.
        Otherwise the MaskFunc is evaluated at runtime

    mask_b: str or MaskFunc
        If a string is provided, the mask specified by the string is retrieved.
        Otherwise the MaskFunc is evaluated at runtime

    Returns
    -------
    MaskFunc
    """

    def f(masker, cube, **kwargs):
        a = (
            mask_a(masker, cube, **kwargs)
            if callable(mask_a)
            else masker.get_mask(mask_a)
        )
        b = (
            mask_b(masker, cube, **kwargs)
            if callable(mask_b)
            else masker.get_mask(mask_b)
        )
        return np.logical_or(a, b)

    return f


def get_default_sftlf_cube():
    """Load NetCDF-SCM's default (last resort) surface land fraction cube"""
    return iris.load_cube(os.path.join(os.path.dirname(__file__), _DEFAULT_SFTLF_FILE))


def get_land_mask(masker, cube, sftlf_cube=None, land_mask_threshold=50, **kwargs):
    """
    Get the land mask

    Parameters
    ----------
    TODO: write this

    Returns
    -------
    np.ndarray
    """
    warn_msg = "Land surface fraction (sftlf) data not available, using default instead"
    sftlf_data = None
    try:
        sftlf_cube = cube.get_metadata_cube(cube.sftlf_var, cube=sftlf_cube)
        sftlf_data = sftlf_cube.cube.data
    except OSError:
        logger.warning(warn_msg)
        def_cube_regridded = get_default_sftlf_cube().regrid(
            cube.cube, iris.analysis.Linear()
        )
        sftlf_data = def_cube_regridded.data

    land_mask = np.where(
        sftlf_data > land_mask_threshold,
        False,  # where it's land, return False i.e. don't mask
        True,  # otherwise True
    )

    return broadcast_onto_lat_lon_grid(cube, land_mask)


def get_nh_mask(masker, cube, **kwargs):
    """
    Get a mask of the Northern Hemisphere

    Parameters
    ----------
    masker : :obj:`CubeMasker`
        Cube masker from which to retrieve the mask

    cube : :obj:`SCMCube`
        Cube to create a mask for

    Returns
    -------
    :obj:`np.ndarray`
        Array of booleans which can be used for the mask
    """
    mask_nh_lat = np.array(
        [cell < 0 for cell in cube.cube.coord(cube.lat_name).cells()]
    )
    mask_all_lon = np.full(cube.cube.coord(cube.lon_name).points.shape, False)

    # Here we make a grid which we can use as a mask. We have to use all
    # of these nots so that our product (which uses AND logic) gives us
    # False in the NH and True in the SH (another way to think of this is
    # that we have to flip everything so False goes to True and True goes
    # to False, do all our operations with AND logic, then flip everything
    # back).
    mask_nh = ~np.outer(~mask_nh_lat, ~mask_all_lon)

    return broadcast_onto_lat_lon_grid(cube, mask_nh)


def get_area_mask(lower_lat, left_lon, upper_lat, right_lon):
    """
    Mask a subset of the globe using latitudes and longitudes in degrees East

    Circular coordinates (longitude) can cross the 0E.

    Parameters
    ----------
    lower_lat : int or float
        Lower latitude bound (degrees North)

    left_lon : int or float
        Lower longitude bound (degrees East)

    upper_lat : int or float
        Upper latitude bound (degrees North)

    right_lon : int or float
        Upper longitude bound (degrees East)

    Returns
    -------
    MaskFunc
    """

    def f(masker, cube, **kwargs):
        def mask_dim(dim, lower, upper):
            # Finds any cells where the bounds overlaps with the range (lower, upper)
            if dim.circular:
                max_val = dim.contiguous_bounds()[-1]
                if lower % max_val > upper % max_val:
                    # Handle case where it wraps around 0
                    return ~np.array(
                        [
                            (lower % max_val < cell).any()
                            or (cell < upper % max_val).any()
                            for cell in dim.bounds
                        ]
                    )
                else:
                    return ~np.array(
                        [
                            (lower % max_val < cell).any()
                            and (cell < upper % max_val).any()
                            for cell in dim.bounds
                        ]
                    )

            else:
                return ~np.array(
                    [
                        (lower < cell).any() and (cell < upper).any()
                        for cell in dim.bounds
                    ]
                )

        mask_lat = mask_dim(cube.lat_dim, lower_lat, upper_lat)
        mask_lon = mask_dim(cube.lon_dim, left_lon, right_lon)

        # Here we make a grid which we can use as a mask. We have to use all
        # of these nots so that our product (which uses AND logic) gives us
        # False in the NH and True in the SH (another way to think of this is
        # that we have to flip everything so False goes to True and True goes
        # to False, do all our operations with AND logic, then flip everything
        # back).
        mask = ~np.outer(~mask_lat, ~mask_lon)

        return broadcast_onto_lat_lon_grid(cube, mask)

    return f


def get_world_mask(masker, cube, **kwargs):
    """
    Get a mask with no values masked out

    Parameters
    ----------
    masker : :obj:`CubeMasker`
        Cube masker from which to retrieve the mask

    cube : :obj:`SCMCube`
        Cube to create a mask for

    Returns
    -------
    :obj:`np.ndarray`
        Array of booleans which can be used for the mask
    """
    return np.full(masker.get_mask("World|Northern Hemisphere").shape, False)


"""dict: known masks"""
MASKS = {
    "World": get_world_mask,
    "World|Northern Hemisphere": get_nh_mask,
    "World|Southern Hemisphere": invert("World|Northern Hemisphere"),
    "World|Land": get_land_mask,
    "World|Ocean": invert("World|Land"),
    "World|Northern Hemisphere|Land": or_masks(
        "World|Northern Hemisphere", "World|Land"
    ),
    "World|Southern Hemisphere|Land": or_masks(
        "World|Southern Hemisphere", "World|Land"
    ),
    "World|Northern Hemisphere|Ocean": or_masks(
        "World|Northern Hemisphere", "World|Ocean"
    ),
    "World|Southern Hemisphere|Ocean": or_masks(
        "World|Southern Hemisphere", "World|Ocean"
    ),
    "World|North Atlantic Ocean": or_masks(get_area_mask(0, -80, 65, 0), "World|Ocean"),
}


class CubeMasker:
    """
    Computes masks for a given cube in a somewhat efficient manner.

    Previously calculated masks are cached so each mask is only calculated once. This
    implementation trades off some additional memory overhead for the ability to
    generate arbitary masks.

    **Adding new masks**

    Additional masks can be added to the MASKS array above. The values in the
    ``MASKS`` array should be MaskFunc's. A MaskFunc is a function which takes a
    ScmCube, CubeMasker and any additional keyword arguments. The function should
    return a numpy array of boolean's with the same dimensionality as the ScmCube.
    Where True values are returned, data will be masked out (excluded) from any
    calculations. The "World|Northern Hemisphere|Land" mask should be True everywhere
    except for land cells in the Northern Hemisphere.

    These MaskFunc's can be composed together to create more complex functionality.
    For example `or_masks(get_area_mask(0, -80, 65, 0), "World|Ocean")` will the
    return the result of an 'or' operation between a subsetted area and an ocean mask.
    """

    def __init__(self, cube, **kwargs):
        """
        Initialise

        Parameters
        ----------
        cube : :obj:`ScmCube`
            cube to generate masks for

        kwargs : dict
            Any optional arguments to be passed to the MaskFunc's during evaluation.
            Possible parameters include:

                sftlf_cube : ScmCube

                land_mask_threshold : float default: 50.
        """
        self.cube = cube
        self._masks = {}
        self.kwargs = kwargs

    def get_mask(self, mask_name):
        """
        Get a single mask

        If the mask has previously been calculated the precalculated result is returned from the cache. Otherwise the appropriate
        MaskFunc is called with any kwargs specified in the constructor.

        Parameters
        ----------
        mask_name : str

        Raises
        ------
        InvalidMask
            If the requested mask cannot be found or evaluated.

        Returns
        -------
        ndarray[bool]
            Any True values should be masked out and excluded from any further calculation.
        """
        try:
            return self._masks[mask_name]
        except KeyError:
            try:
                mask_func = MASKS[mask_name]
                mask = mask_func(self, self.cube, **self.kwargs)
                self._masks[mask_name] = mask
            except KeyError:
                raise InvalidMask("Unknown mask: {}".format(mask_name))

        return mask

    def get_masks(self, mask_names):
        """
        Get a number of masks

        Parameters
        ----------
        mask_names: list of str
            List of masks to attempt to load/calculate.

        Returns
        -------
        dict
            Dictionary where keys are mask names and values are :obj:`np.ndarray` of
            bool

        The result only contains valid masks. Any invalid masks are dropped.
        """
        masks = {}
        for name in mask_names:
            try:
                mask = self.get_mask(name)
                masks[name] = mask
            except InvalidMask as e:
                logger.warning("Failed to create %s mask: %s", name, str(e))
        return masks
