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


def _init_commands():
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
            v = getattr(mod, m)
            if issubclass(v, BaseCommand):
                logger.debug('Found command {}'.format('.'.join([__package__, mod_name, m])))
                cmds.append(v())
    return cmds


# Cache the loaded modules
_commands = _init_commands()


def get_commands():
    return _commands


def run_command(cmd_name, args):
    for cmd in get_commands():
        if cmd.name == cmd_name:
            cmd.run(args)
            logger.debug('command: {} completed successfully'.format(cmd_name))
            return
    raise KeyError('No command {}'.format(cmd_name))
