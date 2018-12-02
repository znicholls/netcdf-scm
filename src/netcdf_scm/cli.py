import argparse
import logging

logger = logging.getLogger('netcdf_scm')


def process_args():
    parser = argparse.ArgumentParser(prog='netcdf-scm',
                                     description='Python wrapper for processing netCDF files for use with simple '
                                                 'climate models')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    return parser.parse_args()


def main():
    _args = process_args()
    root_logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    root_logger.level = logging.DEBUG if _args.verbose else logging.INFO

if __name__ == '__main__':
    main()
