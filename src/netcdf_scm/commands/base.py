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

    def run(self, args):
        pass
