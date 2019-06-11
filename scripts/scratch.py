import os
import os.path

from openscm.scmdataframe import ScmDataFrame
from pymagicc.io import MAGICCData

IN_DIR = "/data/marble/sandbox/share/cmip6-crunched/netcdf-scm-crunched/CMIP6"
OUT_DIR = "/data/marble/sandbox/share/cmip6-wrangled-ipsl-sandbox"

REGIONMODES = {
    "FOURBOX": [
        "World|Northern Hemisphere|Land",
        "World|Northern Hemisphere|Ocean",
        "World|Southern Hemisphere|Land",
        "World|Southern Hemisphere|Ocean",
    ],
    "GLOBAL": ["World"]
}

VARIABLE_MAPPING = {
    "tas": "Surface Temperature",
}

ignored_vars = []

for i, (path, folders, files) in enumerate(os.walk(IN_DIR)):
    if "IPSL" not in path:
        continue
    if files: 
        assert len(files) == 1, files
         
        pb = files[0].split("_")
        variable = pb[1]
        try:
            VARIABLE_MAPPING[variable]
        except KeyError:
            if variable not in ignored_vars:
                print("ignoring {}".format(variable))
                ignored_vars.append(variable)
            continue

        exp_id = pb[4]
        source_id = pb[3]
        variant = pb[5]
        time_id = pb[7].replace(".csv", "")
        base_name = "_".join([variable, exp_id, source_id, variant, time_id])

        d = MAGICCData(ScmDataFrame(os.path.join(path, files[0]))).timeseries()
        d = d.subtract(d.iloc[:, 0], axis='rows')

        print("write .MAG file here")

        for r, regions in REGIONMODES.items():
            m = MAGICCData(d).filter(region=regions)

            m["todo"] = "SET"
            m["variable"] = VARIABLE_MAPPING[variable]

            m.metadata = {"header": "testing file only"}
            try:
                m = m.filter(month=12)
            except TypeError:
                print("Not happy {}".format(files[0]))
                continue

            out_name = "{}_{}_SURFACE_TEMP.IN".format(base_name, r).upper()
            print("wrote {}".format(out_name))
            m.write(os.path.join(OUT_DIR, out_name), magicc_version=7)

print("ignored variables:\n{}".format("\n".join(ignored_vars)))

