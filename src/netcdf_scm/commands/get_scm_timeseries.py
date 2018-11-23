from .base import InOutCommand


class GetSCMTimeseriesCommand(InOutCommand):
    name = 'get_timeseries'

    def run(self, **kwargs):
        cube = super(GetSCMTimeseriesCommand, self).run(**kwargs)

        cube.get_scm_timeseries()

