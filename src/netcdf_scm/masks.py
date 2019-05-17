"""Region definitions
"""

try:
    from iris.util import broadcast_to_shape
except ModuleNotFoundError:
    from .errors import raise_no_iris_warning

    raise_no_iris_warning()

import numpy as np
import warnings

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


def broadcast_onto_lat_lon_grid(cube, array_in):
    """Broadcast an array onto the latitude-longitude grid of ``cube``.

    Here, broadcasting means taking the array and 'duplicating' it so that it
    has the same number of dimensions as the cube's underlying data. For example,
    if our cube has a time dimension of length 3, a latitude dimension of length 4
    and a longitude dimension of length 2 then if we are given in a 4x2 array, we
    broadcast this onto a 3x4x2 array where each slice in the broadcasted array's
    time dimension is identical to the input array.
    """
    lat_length = len(cube.lat_dim.points)
    lon_length = len(cube.lon_dim.points)

    dim_order = [cube.lat_dim_number, cube.lon_dim_number]
    base_shape = (lat_length, lon_length)
    if array_in.shape != base_shape:
        array_in = np.transpose(array_in)

    shape_assert_msg = (
        "the sftlf_cube data must be the same shape as the "
        "cube's longitude-latitude grid"
    )
    assert array_in.shape == base_shape, shape_assert_msg

    return broadcast_to_shape(array_in, cube.cube.shape, dim_order)


class InvalidMask(Exception):
    pass


class CubeMasker:

    def __init__(self, cube, **kwargs):
        self.cube = cube
        self._masks = {}
        self.kwargs = kwargs

        self._mask_funcs = {
            "World": lambda: np.full(self.get_mask("World|Northern Hemisphere").shape, False),
            "World|Northern Hemisphere": self.get_nh_mask,
            "World|Southern Hemisphere": lambda: ~self.get_mask('World|Northern Hemisphere'),
            "World|Land": lambda: self.get_land_mask(**self.kwargs),
            "World|Ocean": lambda: ~self.get_mask('World|Land'),
            "World|Northern Hemisphere|Land": lambda: self.get_hemi_mask("World|Northern Hemisphere", "World|Land"),
            "World|Southern Hemisphere|Land": lambda: self.get_hemi_mask("World|Southern Hemisphere", "World|Land"),
            "World|Northern Hemisphere|Ocean": lambda: self.get_hemi_mask("World|Northern Hemisphere", "World|Ocean"),
            "World|Southern Hemisphere|Ocean": lambda: self.get_hemi_mask("World|Southern Hemisphere", "World|Ocean"),
        }

    def get_mask(self, mask_name):
        try:
            return self._masks[mask_name]
        except KeyError:
            try:
                mask = self._mask_funcs[mask_name]()
                self._masks[mask_name] = mask
            except KeyError:
                raise InvalidMask('Unknown mask: {}'.format(mask_name))
            except InvalidMask:
                raise

        return mask

    def get_masks(self, mask_names):
        masks = {}
        for name in mask_names:
            try:
                mask = self.get_mask(name)
                masks[name] = mask
            except InvalidMask:
                pass
        return masks

    def get_hemi_mask(self, hemi=None, surface=None):
        hemi = self.get_mask(hemi) if hemi is not None else True
        surface = self.get_mask(surface) if surface is not None else True

        return np.logical_or(hemi, surface)

    def get_land_mask(self, sftlf_cube=None, land_mask_threshold=50, **kwargs):
        """Get the land mask.

        Returns
        -------
        np.ndarray
        """
        # Lazy loaded to avoid cyclic dependency
        from .iris_cube_wrappers import SCMCube
        if sftlf_cube is None:
            try:
                sftlf_cube = self.cube.get_metadata_cube(self.cube.sftlf_var)
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

        return broadcast_onto_lat_lon_grid(self.cube, land_mask)

    def get_nh_mask(self):
        mask_nh_lat = np.array(
            [cell < 0 for cell in self.cube.cube.coord(self.cube.lat_name).cells()]
        )
        mask_all_lon = np.full(self.cube.cube.coord(self.cube.lon_name).points.shape, False)

        # Here we make a grid which we can use as a mask. We have to use all
        # of these nots so that our product (which uses AND logic) gives us
        # False in the NH and True in the SH (another way to think of this is
        # that we have to flip everything so False goes to True and True goes
        # to False, do all our operations with AND logic, then flip everything
        # back).
        mask_nh = ~np.outer(~mask_nh_lat, ~mask_all_lon)

        return broadcast_onto_lat_lon_grid(self.cube, mask_nh)