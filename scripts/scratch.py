import datetime as dt
import os
import os.path

import pymagicc
from openscm.scmdataframe import ScmDataFrame
from pymagicc.io import MAGICCData

import netcdf_scm

IN_DIR = "/data/marble/sandbox/share/cmip6-crunched/netcdf-scm-crunched/CMIP6"
OUT_DIR = "/data/marble/sandbox/share/cmip6-wrangled-ipsl-sandbox"

REGIONMODES = {
    "FOURBOX": [
        "World|Northern Hemisphere|Land",
        "World|Northern Hemisphere|Ocean",
        "World|Southern Hemisphere|Land",
        "World|Southern Hemisphere|Ocean",
    ],
    "GLOBAL": ["World"],
}

VARIABLE_MAPPING = {"tas": "Surface Temperature"}

ignored_vars = []

for i, (path, folders, files) in enumerate(os.walk(IN_DIR)):
    if "IPSL" not in path:
        continue
    if files:
        assert len(files) == 1, files

        pb = files[0].split("_")
        variable = pb[1]
        exp_id = pb[4]
        source_id = pb[3]
        variant = pb[5]
        time_id = pb[7].replace(".csv", "")
        base_name = "_".join([variable, exp_id, source_id, variant, time_id])

        d = MAGICCData(ScmDataFrame(os.path.join(path, files[0]))).timeseries()
        d = d.subtract(d.iloc[:, 0], axis="rows")

        mag_name = "{}.MAG".format(base_name).upper()
        print(mag_name)
        mag_writer = MAGICCData(d)
        mag_writer["todo"] = "SET"
        mag_writer.metadata["timeseriestype"] = "MONTHLY"
        mag_writer.metadata["header"] = (
            "Date: {}\n"
            "Crunched by: Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>\n"
            "Writer: pymagicc v{} (available at github.com/openclimatedata/pymagicc)\n"
            "Cruncher: netcdf-scm v{} (available at github.com/znicholls/netcdf-scm)\n"
            "Affiliation: Climate & Energy College, The University of Melbourne, Australia".format(
                dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                pymagicc.__version__,
                netcdf_scm.__version__,
            )
        )
        mag_writer.metadata["other metadata"] = "will go here"
        mag_writer["variable"] = variable  # TODO: fix cruncher so it uses cf names
        try:
            mag_writer.write(os.path.join(OUT_DIR, mag_name), magicc_version=7)
        except TypeError:
            print("Not happy: {}".format(files[0]))
            continue

        try:
            VARIABLE_MAPPING[variable]
        except KeyError:
            if variable not in ignored_vars:
                print("not writing in files for {}".format(variable))
                ignored_vars.append(variable)
            continue

        for r, regions in REGIONMODES.items():
            m = MAGICCData(d).filter(region=regions)

            m["todo"] = "SET"
            m["variable"] = VARIABLE_MAPPING[variable]

            m.metadata = {"header": "testing file only"}
            try:
                m = m.filter(month=12)
            except KeyError:
                print("Not happy {}".format(files[0]))
                continue

            out_name = "{}_{}_SURFACE_TEMP.IN".format(base_name, r).upper()
            print("wrote {}".format(out_name))
            try:
                m.write(os.path.join(OUT_DIR, out_name), magicc_version=7)
            except (TypeError, IndexError):
                print("Not happy {}".format(files[0]))


print("did not write IN files for:\n{}".format("\n".join(ignored_vars)))
