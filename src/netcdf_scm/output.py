"""
Module for handling crunching output tracking

This module handles checking whether a file has already been crunched and if its
source files have been updated since it was last crunched.
"""
import json
import logging
from collections import OrderedDict
from datetime import datetime
from os.path import exists, join

logger = logging.getLogger(__name__)


class OutputFileDatabase:
    """
    Holds a list of output files which have been written.

    Also keeps track of the source files used to create each output file.
    """

    filename = "netcdf-scm_crunched.jsonl"

    def __init__(self, out_dir):
        """
        Initialise.

        Parameters
        ----------
        out_dir : str
            Directory in which to save the database (filename is given by
            ``self.filename``)
        """
        self.out_dir = out_dir
        # Choosing a OrderedDict because it's time complexity for checking if an item
        # already exists is constant, while being able to keep the items in time order
        self._data = OrderedDict()
        self._fp = self.load_from_file()

    def __len__(self):
        """Get length of database"""
        return len(self._data)

    def load_from_file(self):
        """
        Load database from ``self.out_dir``

        Returns
        -------
        :obj:`io.TextIOWrapper`
            Handle to the loaded filepath

        Raises
        ------
        ValueError
            The loaded file contains more than one entry for a given filename
        """
        fname = join(self.out_dir, self.filename)
        if not exists(fname):
            logger.warning("No output tracking file available. Creating new file")
            return open(fname, "w")

        fp = open(fname, "r+")
        lines = fp.readlines()
        for l in lines:
            info = json.loads(l)
            k = info["filename"]
            if k in self._data:
                raise ValueError(
                    "Corrupted output file: duplicate entries for {}".format(k)
                )
            self._data[info["filename"]] = info

        logger.info("Read in %s items from database %s", len(self._data), self.filename)
        return fp

    def register(self, out_fname, info):
        """
        Register a filepath with info in the database

        Parameters
        ----------
        out_fname : str
            Filepath to register

        info : dict
            ``out_fname``'s metadata
        """
        if out_fname in self._data:
            # Need to dump the new order of the contents to file
            del self._data[out_fname]
            self.dump()

        r = {
            **info,
            **{"filename": out_fname, "updated_at": datetime.utcnow().isoformat()},
        }
        self._data[out_fname] = r
        self._write_line(r)
        self._fp.flush()

    def _write_line(self, line):
        """Flush out a line to file"""
        self._fp.write("{}\n".format(json.dumps(line)))

    def dump(self):
        """Rewrite the entire file"""
        logger.info("Rewriting output file")
        self._fp.close()
        # Create a new file truncating the old values
        self._fp = open(join(self.out_dir, self.filename), "w")
        for _, l in self._data.items():
            self._write_line(l)
        self._fp.flush()

    def contains_file(self, filepath):
        """
        Return whether a filepath exists in the database

        Parameters
        ----------
        filepath : str
            Filepath to check (use absolute paths to be safe)

        Returns
        -------
        bool
            If the file is in the database, True, otherwise False
        """
        return filepath in self._data
