import argparse
import logging
import sys

from netcdf_scm.commands import run_command, initialise_parser

logger = logging.getLogger('netcdf_scm')


def process_args():
    parser = argparse.ArgumentParser(prog='netcdf-scm',
                                     description='Python wrapper for processing netCDF files for use with simple '
                                                 'climate models')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    # Add all the parser information from the commands
    subparsers = parser.add_subparsers(dest='cmd')
    initialise_parser(subparsers)

    # Extract the cli arguments
    args = parser.parse_args()

    # If a command is not specified exit with code 1
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    return {k: v for k, v in args._get_kwargs()}


def main():
    args = process_args()
    root_logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    root_logger.level = logging.DEBUG if args['verbose'] else logging.INFO

    # Run the requested command
    cmd_name = args.pop('cmd')
    run_command(cmd_name, args)


if __name__ == '__main__':
    main()
