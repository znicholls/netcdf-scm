"""Masking regions of the World
"""
import warnings

import numpy as np

from netcdf_scm.utils import broadcast_onto_lat_lon_grid

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
    pass


def invert(mask_to_invert):
    def f(masker, cube, **kwargs):
        return ~masker.get_mask(mask_to_invert)

    return f


def or_masks(mask_a, mask_b):
    def f(masker, cube, **kwargs):
        a = mask_a(masker, cube, **kwargs) if callable(mask_a) else masker.get_mask(mask_a)
        b = mask_b(masker, cube, **kwargs) if callable(mask_b) else masker.get_mask(mask_b)
        return np.logical_or(a, b)

    return f


def get_land_mask(masker, cube, sftlf_cube=None, land_mask_threshold=50, **kwargs):
    """Get the land mask.

    Returns
    -------
    np.ndarray
    """
    # Lazy loaded to avoid cyclic dependency
    from .iris_cube_wrappers import SCMCube
    if sftlf_cube is None:
        try:
            sftlf_cube = cube.get_metadata_cube(cube.sftlf_var)
        except OSError:
            warn_msg = (
                "Land surface fraction (sftlf) data not available, only returning "
                "global and hemispheric masks."
            )
            warnings.warn(warn_msg)
            raise InvalidMask()

    if not isinstance(sftlf_cube, SCMCube):
        raise TypeError("sftlf_cube must be an SCMCube instance")

    sftlf_data = sftlf_cube.cube.data

    land_mask = np.where(
        sftlf_data > land_mask_threshold,
        False,  # where it's land, return False i.e. don't mask
        True,  # otherwise True
    )

    return broadcast_onto_lat_lon_grid(cube, land_mask)


def get_nh_mask(masker, cube, **kwargs):
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
    def f(masker, cube, **kwargs):
        def mask_dim(dim, lower, upper):
            # Finds any cells where the bounds overlaps with the range (lower, upper)
            if dim.circular:
                max_val = dim.contiguous_bounds()[-1]
                if lower % max_val > upper % max_val:
                    # Handle case where it wraps around 0
                    return ~np.array(
                        [(lower % max_val < cell).any() or (cell < upper % max_val).any() for cell in dim.bounds]
                    )
                else:
                    return ~np.array(
                        [(lower % max_val < cell).any() and (cell < upper % max_val).any() for cell in dim.bounds]
                    )

            else:
                return ~np.array(
                    [(lower < cell).any() and (cell < upper).any() for cell in dim.bounds]
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
    return np.full(masker.get_mask("World|Northern Hemisphere").shape, False)


MASKS = {
    "World": get_world_mask,
    "World|Northern Hemisphere": get_nh_mask,
    "World|Southern Hemisphere": invert("World|Northern Hemisphere"),
    "World|Land": get_land_mask,
    "World|Ocean": invert("World|Land"),
    "World|Northern Hemisphere|Land": or_masks("World|Northern Hemisphere", "World|Land"),
    "World|Southern Hemisphere|Land": or_masks("World|Southern Hemisphere", "World|Land"),
    "World|Northern Hemisphere|Ocean": or_masks("World|Northern Hemisphere", "World|Ocean"),
    "World|Southern Hemisphere|Ocean": or_masks("World|Southern Hemisphere", "World|Ocean"),
    "World|North Atlantic|Ocean": or_masks(get_area_mask(0, -80, 65, 0), "World|Ocean")
}


class CubeMasker:
    """
    Computes masks for a given cube in a somewhat efficient manner.

    Previously calculated masks are cached so each mask is only calculated once. This implementation trades off some additional
    memory overhead for the ability to generate arbitary masks.

    Adding new masks
    ----------------

    Additional masks can be added to the MASKS
    """

    def __init__(self, cube, **kwargs):
        self.cube = cube
        self._masks = {}
        self.kwargs = kwargs

    def get_mask(self, mask_name):
        try:
            return self._masks[mask_name]
        except KeyError:
            try:
                mask_func = MASKS[mask_name]
                mask = mask_func(self, self.cube, **self.kwargs)
                self._masks[mask_name] = mask
            except KeyError:
                raise InvalidMask('Unknown mask: {}'.format(mask_name))
            except InvalidMask:
                raise

        return mask

    def get_masks(self, mask_names):
        """
        Get arbitary masks

        Parameters
        ----------
        mask_names: list of str
            The masks to calculate

        Returns
        -------
        dict of ndarrays of bool

        The result only contains valid masks. Any invalid masks are dropped.
        """
        masks = {}
        for name in mask_names:
            try:
                mask = self.get_mask(name)
                masks[name] = mask
            except InvalidMask:
                pass
        return masks
