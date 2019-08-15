"""
Module which calculates the weights to be used when taking SCM-box averages

This typically requires considering both the fraction of each cell which is of the desired type (e.g. land or ocean) and the area of each cell. The combination of these two pieces of information creates the weights for each cell which are used when taking area-weighted means.
"""
import logging
import os
from functools import lru_cache

import numpy as np

from ..utils import broadcast_onto_lat_lon_grid

try:
    import iris
    from iris.analysis.cartography import wrap_lons
    from iris.util import broadcast_to_shape
except ModuleNotFoundError:  # pragma: no cover # emergency valve
    from ..errors import raise_no_iris_warning

    raise_no_iris_warning()


logger = logging.getLogger(__name__)

_DEFAULT_SFTLF_FILE = "default_land_ocean_weights.nc"
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


class InvalidWeight(Exception):
    """
    Raised when a weight cannot be calculated.

    This error usually propogates. For example, if a child weight used in the
    calculation of a parent weight fails then the parent weight should also raise an
    InvalidWeight exception (unless it can be satisfactorily handled).
    """


def subtract_weights(weights_to_subtract, subtract_from):
    """
    Subtract weights from some other number

    e.g. useful to convert e.g. from fraction of land to ocean (where
    ocean fractions are 100 - land fractions)

    Parameters
    ----------
    weights_to_subtract: str
        Name of the weights to subtract. These weights are loaded at evaluation time.

    subtract_from : float
        The number from which to subtract the values of ``weights_to_invert`` (once
        loaded)

    Returns
    -------
    :func:`WeightFunc`
        WeightFunc which subtracts the input weights from ``subtract_from``
    """

    def f(weight_calculator, cube, **kwargs):  # pylint:disable=unused-argument
        return (
            subtract_from
            - weight_calculator.get_weights_array_without_area_weighting(
                weights_to_subtract
            )
        )

    return f


def multiply_weights(weight_a, weight_b):
    """
    Take the product of two weights

    Parameters
    ----------
    weight_a : str or WeightFunc
        If a string is provided, the weights specified by the string are retrieved.
        Otherwise the WeightFunc is evaluated at runtime

    weight_b: str or WeightFunc
        If a string is provided, the weights specified by the string are retrieved.
        Otherwise the WeightFunc is evaluated at runtime

    Returns
    -------
    :func:`WeightFunc`
        WeightFunc which multiplies the input weights
    """

    def f(weight_calculator, cube, **kwargs):
        a = (
            weight_a(weight_calculator, cube, **kwargs)
            if callable(weight_a)
            else weight_calculator.get_weights_array_without_area_weighting(weight_a)
        )
        b = (
            weight_b(weight_calculator, cube, **kwargs)
            if callable(weight_b)
            else weight_calculator.get_weights_array_without_area_weighting(weight_b)
        )
        return a * b

    return f


@lru_cache(maxsize=1)
def get_default_sftlf_cube():
    """Load NetCDF-SCM's default (last resort) surface land fraction cube"""
    return iris.load_cube(os.path.join(os.path.dirname(__file__), _DEFAULT_SFTLF_FILE))


def get_land_weights(  # pylint:disable=unused-argument
    weight_calculator, cube, sftlf_cube=None, **kwargs
):
    """
    Get the land weights

    The weights are always adjusted to have units of percentage. If the units are
    detected to be fraction rather than percentage, they will be automatically
    adjusted and a warning will be thrown.

    If the default sftlf cube is used, it is regridded onto ``cube``'s grid using a
    linear interpolation. We hope to use an area-weighted regridding in future but at
    the moment its performance is not good enough to be put into production (
    approximately 100x slower than the linear interpolation regridding).

    Parameters
    ----------
    weight_calculator : :obj:`CubeWeightCalculator`
        Cube weight calculator from which to retrieve the weights

    cube : :obj:`SCMCube`
        Cube to create weights for

    sftlf_cube : :obj:`SCMCube`
        Cube containing the surface land-fraction data

    kwargs : Any
        Ignored (required for compatibility with ``CubeWeightCalculator``)

    Returns
    -------
    np.ndarray
        Land weights
    """
    sftlf_data = None
    try:
        sftlf_cube = cube.get_metadata_cube(cube.sftlf_var, cube=sftlf_cube)
        sftlf_data = sftlf_cube.cube.data
    except OSError:
        warn_msg = (
            "Land surface fraction (sftlf) data not available, using default instead"
        )
        logger.warning(warn_msg)
        try:
            def_cube_regridded = (
                get_default_sftlf_cube()
                .copy()
                .regrid(
                    cube.cube,
                    iris.analysis.Linear(),  # AreaWeighted() in future but too slow now
                )
            )
        except ValueError:  # pragma: no cover # only required for AreaWeighted() regridding
            logger.warning("Guessing bounds to regrid default sftlf data")
            cube.lat_dim.guess_bounds()
            cube.lon_dim.guess_bounds()
            def_cube_regridded = (
                get_default_sftlf_cube()
                .copy()
                .regrid(
                    cube.cube,
                    iris.analysis.Linear(),  # AreaWeighted() in future but too slow now
                )
            )

        sftlf_data = def_cube_regridded.data

    sftlf_data_max = sftlf_data.max()
    if np.isclose(sftlf_data_max, 1, atol=0.3):
        logger.warning(
            "sftlf data max is %s, multiplying by 100 to convert units to percent",
            sftlf_data_max,
        )
        sftlf_data = sftlf_data * 100

    weight_calculator._weights_no_area_weighting[
        "World|Land"
    ] = sftlf_data  # pylint:disable=protected-access
    return broadcast_onto_lat_lon_grid(cube, sftlf_data)


def get_nh_weights(weight_calculator, cube, **kwargs):  # pylint:disable=unused-argument
    """
    Get weights to only include the Northern Hemisphere

    Parameters
    ----------
    weight_calculator : :obj:`CubeWeightCalculator`
        Cube weight calculator from which to retrieve the weights

    cube : :obj:`SCMCube`
        Cube to create weights for

    kwargs : Any
        Ignored (required for compatibility with ``CubeWeightCalculator``)

    Returns
    -------
    :obj:`np.ndarray`
        Northern hemisphere weights
    """
    weights_nh_lat = np.array([c >= 0 for c in cube.lat_dim.points]).astype(int)
    weights_all_lon = np.ones(cube.lon_dim.points.shape)

    weights_nh = np.outer(weights_nh_lat, weights_all_lon)
    weight_calculator._weights_no_area_weighting[  # pylint:disable=protected-access
        "World|Northern Hemisphere"
    ] = weights_nh
    return broadcast_onto_lat_lon_grid(cube, weights_nh)


def get_weights_for_area(lower_lat, left_lon, upper_lat, right_lon):
    """
    Weights a subset of the globe using latitudes and longitudes (in degrees East)

    Iris' standard behaviour is to include any point whose bounds overlap with
    the given ranges e.g. if the range is (0, 130) then a cell whose bounds were
    (-90, 5) would be included even if its point were -42.5.

    This can be altered with the ``ignore_bounds`` keyword argument to
    ``cube.intersection``. In this case only cells whose points lie within the
    range are included so if the range is (0, 130) then a cell whose bounds were
    (-90, 5) would be excluded if its point were -42.5.

    Here we follow the ``ignore_bounds=True`` behaviour (i.e. only include if
    the point lies within the specified range). If we want to only include the
    cell if the entire box is within a point we're going to need to tweak things.
    Given this isn't available in iris, it seems to be an unusual way to do
    intersection so we haven't implemented it.

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
    :func:`WeightFunc`
        WeightFunc which weights out everything except the specified area
    """

    def f(weight_calculator, cube, **kwargs):  # pylint:disable=unused-argument
        lon_dim_pts = cube.lon_dim.points
        lat_dim_pts = cube.lat_dim.points

        lat_lon_size = (
            cube.cube.shape[cube.lat_dim_number],
            cube.cube.shape[cube.lon_dim_number],
        )

        if len(lat_dim_pts.shape) == 1:
            lat_dim_pts = np.broadcast_to(lat_dim_pts, lat_lon_size[::-1]).T
        if len(lon_dim_pts.shape) == 1:
            lon_dim_pts = np.broadcast_to(lon_dim_pts, lat_lon_size)

        weights_lat = ((lower_lat <= lat_dim_pts) & (lat_dim_pts <= upper_lat)).astype(
            int
        )

        lon_modulus = cube.lon_dim.units.modulus
        lon_min = np.floor(lon_dim_pts.min())
        left_lon_wrapped, right_lon_wrapped = wrap_lons(
            np.array([left_lon, right_lon]), lon_min, lon_modulus
        ).astype(int)
        if left_lon_wrapped <= right_lon_wrapped:
            weights_lon = (left_lon_wrapped <= lon_dim_pts) & (
                lon_dim_pts <= right_lon_wrapped
            )
        else:
            weights_lon = (
                (lon_min <= lon_dim_pts) & (lon_dim_pts <= right_lon_wrapped)
            ) | (
                (left_lon_wrapped <= lon_dim_pts)
                & (lon_dim_pts <= lon_min + lon_modulus)
            )

        weights_lon = weights_lon.astype(int)

        # TODO: make issue in Iris about the fact that ``cube.intersection``'s errors
        # are cryptic
        error_msg = "None of the cube's {} lie within the bounds:\nquery: ({}, {})\ncube points: {}"
        if np.equal(np.sum(weights_lon), 0):
            raise ValueError(
                error_msg.format("latitudes", lower_lat, upper_lat, cube.lat_dim.points)
            )

        if np.equal(np.sum(weights_lat), 0):
            raise ValueError(
                error_msg.format("longitudes", left_lon, right_lon, cube.lon_dim.points)
            )

        weights = weights_lon * weights_lat
        return broadcast_onto_lat_lon_grid(cube, weights)

    return f


def get_world_weights(
    weight_calculator, cube, **kwargs
):  # pylint:disable=unused-argument
    """
    Get weights for the world

    Parameters
    ----------
    weight_calculator : :obj:`CubeWeightCalculator`
        Cube weight calculator from which to retrieve the weights

    cube : :obj:`SCMCube`
        Cube to create weights for

    kwargs : Any
        Ignored (required for compatibility with ``CubeWeightCalculator``)

    Returns
    -------
    :obj:`np.ndarray`
        Weights which can be used for the world mean calculation
    """
    return np.ones(
        weight_calculator.get_weights_array_without_area_weighting(
            "World|Northern Hemisphere"
        ).shape
    )


"""dict: in-built functions to calculate weights for different regions without area weighting"""
WEIGHTS_FUNCTIONS_WITHOUT_AREA_WEIGHTING = {
    "World": get_world_weights,
    "World|Northern Hemisphere": get_nh_weights,
    "World|Southern Hemisphere": subtract_weights("World|Northern Hemisphere", 1),
    "World|Land": get_land_weights,
    "World|Ocean": subtract_weights("World|Land", 100),
    "World|Northern Hemisphere|Land": multiply_weights(
        "World|Northern Hemisphere", "World|Land"
    ),
    "World|Southern Hemisphere|Land": multiply_weights(
        "World|Southern Hemisphere", "World|Land"
    ),
    "World|Northern Hemisphere|Ocean": multiply_weights(
        "World|Northern Hemisphere", "World|Ocean"
    ),
    "World|Southern Hemisphere|Ocean": multiply_weights(
        "World|Southern Hemisphere", "World|Ocean"
    ),
    "World|North Atlantic Ocean": multiply_weights(
        get_weights_for_area(0, -80, 65, 0), "World|Ocean"
    ),
    # 5N-5S, 170W-120W (i.e. 190E to 240E) see
    # https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
    "World|El Nino N3.4": multiply_weights(
        get_weights_for_area(-5, 190, 5, 240), "World|Ocean"
    ),
}


class CubeWeightCalculator:
    """
    Computes weights for a given cube in a somewhat efficient manner.

    Previously calculated weights are cached so each set of weights is only calculated
    once. This implementation trades off some additional memory overhead for the
    ability to generate arbitary weights.

    **Adding new weights**

    Additional weights can be added to ``netcdf_scm.weights.weights``. The values in
    ``weights`` should be WeightFunc's. A WeightFunc is a function which takes a
    ScmCube, CubeWeightCalculator and any additional keyword arguments. The function should
    return a numpy array of boolean's with the same dimensionality as the ScmCube.
    Where True values are returned, data will be weightsed out (excluded) from any
    calculations. For example, the "World|Northern Hemisphere|Land" weights should be
    True everywhere except for land cells in the Northern Hemisphere.

    These WeightFunc's can be composed together to create more complex functionality.
    For example `multiply_weights(get_weights_for_area(0, -80, 65, 0), "World|Ocean")` will
    return the result of an 'or' operation between the given area and an ocean weights.
    """

    def __init__(self, cube, **kwargs):
        """
        Initialise

        Parameters
        ----------
        cube : :obj:`ScmCube`
            cube to generate weights for

        kwargs : dict
            Any optional arguments to be passed to the WeightFunc's during evaluation.
            Possible parameters include:

                sftlf_cube : ScmCube

                land_weights_threshold : float default: 50.
        """
        self.cube = cube
        self._weights_no_area_weighting = {}
        self._weights = {}
        self._area_weights = None
        self.kwargs = kwargs

    def get_weights_array_without_area_weighting(self, weights_name):
        """
        Get a single weights array without any consideration of area weighting

        Parameters
        ----------
        weights_name : str


        """
        try:
            return self._weights_no_area_weighting[weights_name]
        except KeyError:
            try:
                weights_func = WEIGHTS_FUNCTIONS_WITHOUT_AREA_WEIGHTING[weights_name]
                weights = weights_func(self, self.cube, **self.kwargs)
                if len(weights.shape) == 2:
                    # ensure weights can be used directly on cube
                    weights = broadcast_to_shape(
                        weights,
                        self.cube.cube.shape,
                        [self.cube.lat_dim_number, self.cube.lon_dim_number],
                    )

                self._weights_no_area_weighting[weights_name] = weights
            except KeyError:
                raise InvalidWeight("Unknown weights: {}".format(weights_name))

        return weights

    def _get_area_weights(self):
        if self._area_weights is None:
            raw_area_weights = self.cube.get_metadata_cube(
                self.cube.areacella_var
            ).cube.data
            self._area_weights = broadcast_onto_lat_lon_grid(
                self.cube, raw_area_weights
            )

        return self._area_weights

    def get_weights_array(self, weights_name):
        """
        Get a single weights array

        If the weights has previously been calculated the precalculated result is
        returned from the cache. Otherwise the appropriate WeightFunc is called with any
        kwargs specified in the constructor.

        Parameters
        ----------
        weights_name : str

        Raises
        ------
        InvalidWeight
            If the requested weights cannot be found or evaluated

        ValueError
            If the cube has no data which matches the input weights

        Returns
        -------
        ndarray[bool]
            Any True values should be weightsed out and excluded from any further calculation.
        """
        try:
            return self._weights[weights_name]
        except KeyError:
            weights_without_area = self.get_weights_array_without_area_weighting(
                weights_name
            )
            area_weights = self._get_area_weights()
            weights = weights_without_area * area_weights
            self._weights[weights_name] = weights

        if np.equal(np.sum(weights), 0):
            raise ValueError(
                "Your cube has no data which matches the `{}` weights".format(
                    weights_name
                )
            )

        return weights

    def get_weights(self, weights_names):
        """
        Get a number of weights

        Parameters
        ----------
        weights_names: list of str
            List of weights to attempt to load/calculate.

        Returns
        -------
        dict
            Dictionary where keys are weights names and values are :obj:`np.ndarray` of
            bool

        The result only contains valid weights. Any invalid weights are dropped.
        """
        weights = {}
        for name in weights_names:
            try:
                weights_for_name = self.get_weights_array(name)
                weights[name] = weights_for_name
            except InvalidWeight as e:
                logger.warning("Failed to create %s weights: %s", name, str(e))

        return weights
