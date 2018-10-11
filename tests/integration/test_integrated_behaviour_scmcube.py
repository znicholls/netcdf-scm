from unittest.mock import MagicMock

import pytest
import numpy as np
import iris


from netcdf_scm.iris_cube_wrappers import SCMCube, MarbleCMIP5Cube
from conftest import TEST_TAS_FILE, TEST_AREACELLA_FILE


class TestSCMCubeIntegration(object):
    tclass = SCMCube

    def test_load_data_and_areacella(self, test_cube):
        tfile = TEST_TAS_FILE
        test_cube.get_file_from_load_data_args = MagicMock(return_value=tfile)

        test_constraint = iris.Constraint(
            cube_func=(lambda c: c.var_name == np.str("tas"))
        )
        test_cube.get_variable_constraint_from_load_data_args = MagicMock(
            return_value=test_constraint
        )

        tmdata_scmcube = type(test_cube)()
        tmdata_scmcube.cube = iris.load_cube(TEST_AREACELLA_FILE)
        test_cube.get_metadata_cube = MagicMock(return_value=tmdata_scmcube)

        tkwargs = {
            "variable_name": "fco2antt",
            "modeling_realm": "Amon",
            "model": "CanESM2",
            "experiment": "1pctCO2",
        }

        with pytest.warns(None) as record:
            test_cube.load_data(**tkwargs)
        # Gracefully filling warnings, when we move to iris v2.2.0, change this to zero
        # as that bug will be fixed
        assert len(record) == 6

        test_cube.get_file_from_load_data_args.assert_called_with(**tkwargs)
        test_cube.get_variable_constraint_from_load_data_args.assert_called_with(
            **tkwargs
        )
        test_cube.get_metadata_cube.assert_called_with(test_cube._areacella_var)

        cell_measures = test_cube.cube.cell_measures()
        assert len(cell_measures) == 1
        assert cell_measures[0].standard_name == "cell_area"


class TestMarbleCMIP5Cube(TestSCMCubeIntegration):
    tclass = MarbleCMIP5Cube
