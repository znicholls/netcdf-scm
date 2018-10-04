from os import listdir
from os.path import join, splitext, basename, isdir
from datetime import datetime
from copy import deepcopy


import numpy as np
import pandas as pd
import re
import iris
from pymagicc.io import MAGICCData


class SCMCube(object):
    def _get_data_path(self):
        """

        """
        raise NotImplementedError()

    def _get_data_name(self):
        """

        """
        raise NotImplementedError()

    def _get_metadata_cube_info(self):
        """

        """
        raise NotImplementedError()


class MarbleCMIP5Cube(SCMCube):
    pass
