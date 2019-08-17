"""Miscellaneous definitions used in NetCDF-SCM"""
_SCM_TIMESERIES_META_COLUMNS = [
    "variable",
    "variable_standard_name",
    "region",
    "scenario",
    "climate_model",
    "member_id",
    "mip_era",
    "activity_id",
]
"""Metadata columns to include when creating SCM timeseries"""

_LAND_FRACTION_REGIONS = [
    "World",
    "World|Land",
    "World|Ocean",
    "World|Northern Hemisphere",
    "World|Northern Hemisphere|Land",
    "World|Northern Hemisphere|Ocean",
    "World|Southern Hemisphere",
    "World|Southern Hemisphere|Land",
    "World|Southern Hemisphere|Ocean",
]
"""
list: Regions required to perform land fraction calculations

We require all the ocean regions too as it only makes sense to return land fraction if
we're actually looking at data which contains a split.
"""
