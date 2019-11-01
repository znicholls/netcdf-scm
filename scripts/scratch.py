import os.path

from netcdf_scm.cli import _do_wrangling

_do_wrangling(
    os.path.join("./tests/test-data/expected-crunching-output/cmip6output/CMIP6/"),
    os.path.join("output-examples/wrangling-scratch"),
    # "^(?!.*(fx)).*$",
    "^(?!.*(fx)).*tas.*$",
    "mag-files",
    # "mag-files-average-year-mid-year",
    True,
    "Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>",
    "CMIP6Output",
    5,
    None,
)
