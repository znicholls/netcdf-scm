import datetime as dt
import glob
import os

import iris
import pandas as pd
from iris.pandas import as_data_frame
from netcdf_scm.iris_cube_wrappers import SCMCube
from openscm.scmdataframe import df_append
import pymagicc
from pymagicc.io import MAGICCData
from pymagicc.definitions import convert_magicc7_to_openscm_variables

IN_DIR = "/data/marble/FrozenProject_Repository/CMIP6GHGConcentrationProjections_1_2_1"
IN_DIR = "/Users/znicholls/Desktop/AGCEC/Data/CMIP6GHGConcentrationProjections_1_2_1"
IN_DIR_HIST = "/data/marble/FrozenProject_Repository/CMIP6GHGConcentrationHistorical_1_2_0"
IN_DIR_HIST = "/Users/znicholls/Desktop/AGCEC/Data/CMIP6GHGConcentrationHistorical_1_2_0"

OUT_DIR = "/data/marble/sandbox/share/cmip6-wrangled-ipsl-sandbox"
OUT_DIR = "/Users/znicholls/Desktop/scratch-crunch"

REGION_MAPPING = {
    "Global": "World",
    "Northern Hemisphere": "World|Northern Hemisphere",
    "Southern Hemisphere": "World|Southern Hemisphere",
}

UNIT_MAPPING = {
    "1.e-6": "ppm",
    "1.e-9": "ppb",
    "1.e-12": "ppt",
}

IN_REGIONS = [
    "World|Northern Hemisphere",
    "World|Southern Hemisphere",
]
VARIABLE_MAPPINGS = {
    "Atmospheric Concentrations|HFC143A": "Atmospheric Concentrations|HFC143a",
    "Atmospheric Concentrations|METHYL_CHLORIDE": "Atmospheric Concentrations|CH3Cl",
    "Atmospheric Concentrations|HCFC141B": "Atmospheric Concentrations|HCFC141b",
    "Atmospheric Concentrations|C_C4F8": "Atmospheric Concentrations|cC4F8",
    "Atmospheric Concentrations|HFC227EA": "Atmospheric Concentrations|HFC227ea",
    "Atmospheric Concentrations|NITROUS_OXIDE": "Atmospheric Concentrations|N2O",
    "Atmospheric Concentrations|HALON1211": "Atmospheric Concentrations|Halon1211",
    "Atmospheric Concentrations|HFC134A": "Atmospheric Concentrations|HFC134a",
    "Atmospheric Concentrations|METHANE": "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|CARBON_TETRACHLORIDE": "Atmospheric Concentrations|CCl4",
    "Atmospheric Concentrations|CARBON_DIOXIDE": "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|METHYL_BROMIDE": "Atmospheric Concentrations|CH3Br",
    "Atmospheric Concentrations|HFC236FA": "Atmospheric Concentrations|HFC236fa",
    "Atmospheric Concentrations|CH3CCL3": "Atmospheric Concentrations|CH3CCl3",
    "Atmospheric Concentrations|HFC4310MEE": "Atmospheric Concentrations|HFC4310",
    "Atmospheric Concentrations|CH2CL2": "Atmospheric Concentrations|CH2Cl2",
    "Atmospheric Concentrations|HALON1301": "Atmospheric Concentrations|Halon1301",
    "Atmospheric Concentrations|HALON2402": "Atmospheric Concentrations|Halon2402",
    "Atmospheric Concentrations|HFC365MFC": "Atmospheric Concentrations|HFC365mfc",
    "Atmospheric Concentrations|HCFC142B": "Atmospheric Concentrations|HCFC142b",
    "Atmospheric Concentrations|CHCL3": "Atmospheric Concentrations|CHCl3",
    "Atmospheric Concentrations|HFC245FA": "Atmospheric Concentrations|HFC245fa",
    "Atmospheric Concentrations|HFC152A": "Atmospheric Concentrations|HFC152a",
}

skipped = {}
for i, hf in enumerate(glob.glob(os.path.join(IN_DIR_HIST, "*GMNHSH*2014.nc"))):
    print(hf)
    scm_cube = SCMCube()
    scm_cube.cube = iris.load_cube(hf)[1000:, :]

    variable = "Atmospheric Concentrations|{}".format(scm_cube.cube.var_name.replace(
            "mole_fraction_of_",
            ""
        ).replace(
            "_in_air",
            ""
        ).upper()
    )
    if variable in VARIABLE_MAPPINGS:
        variable = VARIABLE_MAPPINGS[variable]

    magicc_var = convert_magicc7_to_openscm_variables(variable, inverse=True)
    if variable == magicc_var or "EQ" in variable:
        print("skipping {}".format(variable))
        skipped[i] = variable
        continue

    scm_cube._adjust_gregorian_year_zero_units()
    scm_cube.cube = scm_cube.cube.extract(
        iris.Constraint(time=lambda x: x.point.year >= 1000)
    )

    hist_df = as_data_frame(scm_cube.cube)
    to_openscm_region_mapping = {
        int(r.split(":")[0].strip()): REGION_MAPPING[r.split(":")[1].strip()]
        for r in scm_cube.cube.coord("sector").attributes["ids"].split(";")
    }
    hist_df.columns = hist_df.columns.map(to_openscm_region_mapping)

    cf_variable = os.path.basename(hf).split("_")[0]

    all_together = []
    for pf in glob.glob(os.path.join(IN_DIR, "{}*GMNHSH*.nc".format(cf_variable))):
        scenario = os.path.basename(pf).split("_")[4]
        scen_cube = iris.load_cube(pf)
        to_openscm_region_mapping = {
            int(r.split(":")[0].strip()): REGION_MAPPING[r.split(":")[1].strip()]
            for r in scen_cube.coord("sector").attributes["ids"].split(";")
        }

        scen_df = as_data_frame(scen_cube)
        scen_df.columns = scen_df.columns.map(to_openscm_region_mapping)

        full_df = pd.concat([hist_df, scen_df])
        full_df.index = full_df.index.map(
            lambda x: dt.datetime(x.year, x.month, x.day, x.hour)
        )

        writer = MAGICCData(
            data=full_df,
            columns={
                "region": full_df.columns,
                "scenario": scenario,
                "model": "unspecified",
                "unit": UNIT_MAPPING[str(scm_cube.cube.units)],
                "variable": variable,
            }
        )
        writer["parameter_type"] = "point"
        writer = writer.interpolate([
            dt.datetime(y, 12, 31) for y in range(1000, 2501)
        ])
        writer["todo"] = "SET"
        writer.metadata = {}
        writer.metadata["header"] = (
            "Date: {}\n"
            "Crunched by: Zebedee Nicholls <zebedee.nicholls@climate-energy-college.org>, Jared Lewis <jared.lewis@climate-energy-college.org>, Malte Meinshausen <malte.meinshausen@unimelb.edu.au>\n"
            "Writer: pymagicc v{} (available at github.com/openclimatedata/pymagicc)\n"
            "Affiliation: Climate & Energy College, The University of Melbourne, Australia\n\n"
            "~~~ HISTORICAL DATA INFO ~~~\n"
            "\n"
            "{}\n"
            "\n"
            "~~~ SCENARIO DATA INFO ~~~\n"
            "\n"
            "{}\n"
            "\n"
            "Note: Data is an annual average, centred on the 31st December of the nominated year\n\n".format(
                dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                pymagicc.__version__,
                "\n".join(["{}: {}".format(k, v) for k, v in scm_cube.cube.attributes.items()]),
                "\n".join(["{}: {}".format(k, v) for k, v in scen_cube.attributes.items()])
            )
        )


        out_file = os.path.join(OUT_DIR, "{}_HEMISPHERIC_{}.IN".format(scenario.upper(), magicc_var))
        print(out_file)
        writer.filter(region=IN_REGIONS).write(out_file, magicc_version=7)

        out_file = os.path.join(OUT_DIR, "{}_GLOBAL_{}.IN".format(scenario.upper(), magicc_var))
        print(out_file)
        writer.filter(region="World").write(out_file, magicc_version=7)

        writer.metadata["further_info"] = "http://climatecollege.unimelb.edu.au/cmip6"
        writer.metadata["esgf_server"] = "https://esgf-node.llnl.gov/projects/input4mips/"
        writer.metadata["timeseriestype"] = "AVERAGE_YEAR_END_OF_YEAR"
        writer.metadata["scenario"] = scenario

        out_file = os.path.join(OUT_DIR, "{}_{}.MAG".format(scenario.upper(), magicc_var))
        print(out_file)
        writer.write(out_file, magicc_version=7)

print("skipped {}".format(skipped))
