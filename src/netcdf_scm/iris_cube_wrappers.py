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
    def load_data(self):
        raise NotImplementedError()

    def _get_data_path(self):
        """
        Get the path to a data file from self's attributes.

        This can take multiple forms, it may just return a previously set
        filepath attribute or it could combine a number of different metadata
        elements (e.g. model name, experiment name) to create the data path.
        """
        raise NotImplementedError()

    def _get_data_name(self):
        """
        Get the name of a data file from self's attributes.

        This can take multiple forms, it may just return a previously set
        filename attribute or it could combine a number of different metadata
        elements (e.g. model name, experiment name) to create the data name.
        """
        raise NotImplementedError()

    def _get_metadata_load_arguments(self, variable):
        """
        Get the name of a metadata file from self's attributes.

        This can take multiple forms, it may just return a previously set
        metada_filename attribute or it could combine a number of different
        metadata elements (e.g. model name, experiment name) to create the
        metadata filename.

        # Parameters
        variable (str): the name of the metadata variable to get, as it
            appears in the filename.

        # Returns
        load_args (dict): dictionary containing all the arguments to pass to
            `self.load_data` required to load the desired metadata cube.
        """
        raise NotImplementedError()

    def get_metadata_cube(self, variable):
        """
        Load a metadata cube from self's attributes.

        # Parameters
        variable (str): the name of the metadata variable to get, as it
            appears in the filename.
        """
        load_args = self._get_metadata_load_arguments(variable)

        metadata_cube = type(self)()
        metadata_cube.load_data(**load_args)

        return metadata_cube


class MarbleCMIP5Cube(_SCMCube):
    """
    Subclass of `_SCMCube` which can be used with the `cmip5` directory on marble

    This directory structure is very similar, but not quite identical, to the
    recommended CMIP5 directory structure.
    """
    pass
