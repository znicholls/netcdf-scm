from os import walk
from os.path import abspath, dirname, join, isfile
import subprocess


import pandas as pd
import numpy as np


from conftest import TEST_DATA_KNMI_DIR, TEST_DATA_MARBLE_CMIP5_DIR


def test_crunching(tmpdir):
    here = abspath(dirname(__file__))

    THRESHOLD_PERCENTAGE_DIFF = 10 ** -1

    SCRIPT_TO_RUN = join(here, "..", "..", "scripts/crunch_to_scm.py")
    INPUT_DIR = TEST_DATA_MARBLE_CMIP5_DIR
    OUTPUT_DIR = tmpdir
    VAR_TO_CRUNCH = "tas"

    command = [
        "python",
        SCRIPT_TO_RUN,
        INPUT_DIR,
        OUTPUT_DIR,
        "--var-to-crunch",
        VAR_TO_CRUNCH,
        "-f",
    ]
    subprocess.check_call(command)

    for dirpath, dirnames, filenames in walk(OUTPUT_DIR):
        if not dirnames:
            assert len(filenames) == 1
            filename = filenames[0]

            knmi_data_name = "global_{}.dat".format("_".join(filename.split("_")[1:6]))
            knmi_data_path = join(TEST_DATA_KNMI_DIR, knmi_data_name)

            if not isfile(knmi_data_path):
                print("No data available for {}".format(knmi_data_path))
                continue

            knmi_data = pd.read_csv(
                knmi_data_path,
                skiprows=3,
                delim_whitespace=True,
                header=None,
                names=["year", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            ).melt(id_vars="year", var_name="month")
            knmi_data["year"] = knmi_data["year"].astype(int)
            knmi_data["month"] = knmi_data["month"].astype(int)
            knmi_data = knmi_data.set_index(["year", "month"])

            crunched_data = (
                pd.read_csv(join(dirpath, filename), header=list(range(5)), index_col=0)
                .reset_index()
                .melt(id_vars="time")
            )

            comparison_data = crunched_data[crunched_data["region"] == "World"][
                ["time", "value"]
            ]
            comparison_data["year"] = comparison_data["time"].apply(
                lambda x: int(x.split("-")[0])
            )
            comparison_data["month"] = comparison_data["time"].apply(
                lambda x: int(x.split("-")[1])
            )
            comparison_data = comparison_data.drop("time", axis="columns")
            comparison_data = comparison_data.set_index(["year", "month"])

            rel_difference = (knmi_data - comparison_data) / knmi_data
            # drop regions where times are not equal
            rel_difference = rel_difference.dropna()

            assert_message = "{} data is not the same to within {}%".format(
                filename, THRESHOLD_PERCENTAGE_DIFF
            )

            assert (
                np.abs(rel_difference.values) < THRESHOLD_PERCENTAGE_DIFF / 100
            ).all(), assert_message
            print(
                "{} file matches KNMI data to within {}%".format(
                    filename, THRESHOLD_PERCENTAGE_DIFF
                )
            )
