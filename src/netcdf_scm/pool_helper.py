from .utils import take_lat_lon_mean

def add_masked_cube(input):
    (
        mask,
        helper,
        sftlf_cube,
        land_mask_threshold,
        land_fraction_regions,
        area_weights
    ) = input
    scm_cube = helper.get_scm_cubes(
                sftlf_cube=sftlf_cube,
                land_mask_threshold=land_mask_threshold,
                masks=[mask],
    )[mask]

    if mask in land_fraction_regions:
        area = helper._get_area(scm_cube, area_weights)
    else:
        area = None
    return mask, take_lat_lon_mean(scm_cube, area_weights), area

def funtime(a):
    return a**2
