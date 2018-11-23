from netcdf_scm.iris_cube_wrappers import MarbleCMIP5Cube


class BaseCommand(object):
    """
    The base class for representing a command which can be run via the commandline.
    """

    # Command name. This is used in the cli to select this command
    name = ''

    # Help string for command
    help = ''

    def initialise_parser(self, subparser):
        pass

    def run(self, **kwargs):
        pass


class InOutCommand(BaseCommand):
    """
    A cube which can load and save cube data
    """

    def initialise_parser(self, subparser):
        subparser.add_argument('--root-dir',
                               help='')
        subparser.add_argument('--activity')
        subparser.add_argument('--experiment')
        subparser.add_argument('--modeling-realm')
        subparser.add_argument('--variable-name')
        subparser.add_argument('--model')
        subparser.add_argument('--ensemble-member')
        subparser.add_argument('--time-period')
        subparser.add_argument('--file-ext')

        # Output args
        subparser.add_argument('--out')
        subparser.add_argument('--out-format')

    def load_cube(self, **kwargs):
        cube = MarbleCMIP5Cube()
        cube.load_data(**kwargs)

        return cube

    def run(self, **kwargs):
        return self.load_cube(**kwargs)
