from unittest.mock import patch, MagicMock

import pytest

from netcdf_scm.commands import run_command
from netcdf_scm.commands.base import BaseCommand

@patch('netcdf_scm.commands.get_commands')
def test_missing_command(mock_commands):
    mock_commands.return_value = []
    with pytest.raises(KeyError):
        run_command('missing', {})


@patch('netcdf_scm.commands.get_commands')
def test_runs_command(mock_commands):
    mock_command = MagicMock(spec=BaseCommand())
    mock_command.name = 'test'
    mock_commands.return_value = [
        mock_command
    ]

    run_command('test', {
        'passed_arg': True
    })

    mock_command.run.assert_called_with({
        'passed_arg': True
    })