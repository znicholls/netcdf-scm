"""This package contains the commands that can be run via the command line tool

This module provides a wrapper for exposing the functionality of netcdf-scm to a commandline tool. This provides a
easy to customise framework for defining more advanced commandline operations which consist of more than one function
call.

See netcdf-scm.cli for more information about the commandline tool.

Each command is defined as a class inheriting the `netcdf-scm.commands.base.BaseCommand` which contains a `run` method
and options for adding any additional command line parameters.
"""
from importlib import import_module
from logging import getLogger
from pkgutil import iter_modules

from .base import BaseCommand

logger = getLogger(__name__)


def _find_commands():
    cmds = []

    # Find all submodules
    for pkg, mod_name, is_pkg in iter_modules(__path__):
        # Import the submodule and iterate over any attributes
        mod = import_module(__package__ + '.' + mod_name)
        for m in dir(mod):
            # Skip any attributes starting with '_'
            if m.startswith('_'):
                continue

            # Check if subclass of BaseCommand
            try:
                v = getattr(mod, m)
                if issubclass(v, BaseCommand) and v.name:
                    logger.debug('Found command {}: {}'.format(v.name, '.'.join([__package__, mod_name, m])))
                    cmds.append(v())
            except TypeError:
                pass
    return cmds


# Cache the loaded modules
_commands = _find_commands()


def get_commands():
    return _commands


def initialise_parser(parser):
    for cmd in get_commands():
        p = parser.add_parser(cmd.name, help=cmd.help)
        cmd.initialise_parser(p)


def run_command(cmd_name, cli_args):
    # Drop all the None args
    kwargs = {k: cli_args[k] for k in cli_args if cli_args[k] is not None}

    for cmd in get_commands():
        if cmd.name == cmd_name:
            cmd.run(**kwargs)
            logger.debug('command: {} completed successfully'.format(cmd_name))
            return
    raise KeyError('No command {}'.format(cmd_name))
