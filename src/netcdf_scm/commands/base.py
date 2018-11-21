class BaseCommand(object):
    """
    The base class for representing a command which can be run via the commandline.
    """

    name = ''

    def initialise(self, subparser):
        pass

    def run(self):
        pass
