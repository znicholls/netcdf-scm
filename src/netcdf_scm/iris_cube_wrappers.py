from os import listdir
from os.path import join, splitext, basename, isdir
from datetime import datetime
from copy import deepcopy


import numpy as np
import pandas as pd
import re
import iris
from pymagicc.io import MAGICCData


class _SCMCube(object):
    """
    Provides the ability to process netCDF files for use in simple climate models.

    This base class contains the most common operations. However to fully
    utilise its power you must use a subclass of it, which defines the methods
    which raise `NotImplementedError`'s in this class.
    """
    def _get_data_path(self):
        """
        Get the path to a data file from the cube's attributes.

        This can take multiple forms, it may just return a previously set
        filepath attribute or it could combine a number of different metadata
        elements (e.g. model name, experiment name) to create the data path.
        """
        raise NotImplementedError()

    def _get_data_name(self):
        """
        Get the name of a data file from the cube's attributes.

        This can take multiple forms, it may just return a previously set
        filename attribute or it could combine a number of different metadata
        elements (e.g. model name, experiment name) to create the data name.
        """
        raise NotImplementedError()

    def _get_metadata_filename(self, variable):
        """
        Get the name of a metadata file from the cube's attributes.

        This can take multiple forms, it may just return a previously set
        metada_filename attribute or it could combine a number of different
        metadata elements (e.g. model name, experiment name) to create the
        metadata filename.

        # Parameters
        variable (str): the name of the variable to get, as it appears in the
            filename.
        """
        raise NotImplementedError()


class MarbleCMIP5Cube(_SCMCube):
    pass
