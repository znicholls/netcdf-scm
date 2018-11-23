from netcdf_scm.commands.base import InOutCommand
from unittest.mock import patch


@patch('netcdf_scm.commands.base.MarbleCMIP5Cube.load_data')
def test_load_data(mock_load):
    tkwargs = {
        "variable_name": "fco2antt",
        "modeling_realm": "Amon",
        "model": "CanESM2",
        "experiment": "1pctCO2",
        "file_ext": ".nc"
    }

    i = InOutCommand()
    i.load_cube(**tkwargs)

    mock_load.assert_called_with(**tkwargs)