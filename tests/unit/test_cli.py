import pytest

from argparse import Namespace
from netcdf_scm.cli import main

from unittest.mock import patch
from logging import getLogger, DEBUG, INFO

# Default cli parameters to include in the config. These are normally set in `netcdf_scm.cli.process_args` from sys.argv
default_config = {
    'verbose': False
}


@pytest.fixture
def with_cli_config():
    """
    Set configuration for the CLI

    Returns
    -------
    Callable function to set the configuration values used by the CLI. These values are a combination of the default
    values (`default_config`) and the kwargs passed to the function.
    """
    with patch('netcdf_scm.cli.process_args') as mock_process_args:
        def set_config(**d):
            args = default_config.copy()
            args.update(d)
            mock_process_args.return_value = Namespace(**args)
        yield set_config


def test_verbose_flag(with_cli_config):
    with_cli_config()
    main()

    assert getLogger().level == INFO

    with_cli_config(verbose=True)
    main()

    assert getLogger().level == DEBUG


