import datetime as dt
import os

import pandas as pd

from pymagicc.io import MAGICCData
from pymagicc.definitions import convert_magicc7_to_openscm_variables

OUT_DIR = "/data/marble/sandbox/share/cmip6-wrangled-ipsl-sandbox"

rcps = [
    "RCP3PD",
    "RCP6",
    "RCP45",
    "RCP85",
]

failures = []
for rcp in rcps:
    print(rcp)
    database = MAGICCData(os.path.join("./", "{}_MIDYEAR_CONCENTRATIONS.DAT".format(rcp)))
    database["parameter_type"] = "point"

    metadata = database.metadata
    data = database.timeseries()

    # calculate averages centred on end of year rather than centred on middle of year
    copy_data = data.copy()
    copy_data[dt.datetime(copy_data.columns[-1].year + 1, copy_data.columns[-1].month, copy_data.columns[-1].day)] = copy_data.iloc[:, -1]
    copy_data = copy_data.iloc[:, 1:]
    copy_data.columns = data.columns
    data = (data + copy_data) / 2

    database = MAGICCData(data) 
    database.metadata = metadata
    database.metadata["header"] = "{}\n{}".format(
        database.metadata["header"],
        "Note: Resampled to provide averages centred on Dec 31 of the given year rather than the middle of the year"
    )
    for v in database["variable"]:
        print(v)
        magicc_var = convert_magicc7_to_openscm_variables(v, inverse=True)
        
        if v == magicc_var:
            print("skipping {}".format(v))
            continue
        print(magicc_var)

        writer = database.filter(variable=v, region="World")
        out_file = os.path.join(OUT_DIR, "{}_{}.IN".format(rcp, magicc_var))
        print(out_file)

        try:
            writer.write(out_file, magicc_version=7)
        except:
            print("failed to write {}".format(v))
            failures.append(out_file)
            continue

print("failures\n{}".format("\n".join(failures)))
