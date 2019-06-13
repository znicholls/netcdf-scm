import os

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
