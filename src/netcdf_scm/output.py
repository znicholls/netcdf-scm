import json
import logging
from collections import OrderedDict
from datetime import datetime
from os.path import exists, join

logger = logging.getLogger(__name__)


class OutputFileDatabase(object):
    """
    Holds a list of output files which have been written.

    Also keeps track of the source files used to create each output file.
    """

    filename = "netcdf-scm_crunched.jsonl"

    def __init__(self, out_dir):
        self.out_dir = out_dir
        # Choosing a OrderedDict because it's time complexity for checking if an item
        # already exists is constant, while being able to keep the items in time order
        self._data = OrderedDict()
        self._fp = self.load_from_file()

    def __len__(self):
        return len(self._data)

    def load_from_file(self):
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
                )  # pragma: no cover # emergency valve
            self._data[info["filename"]] = info

        logger.info("Read in {} items from cru".format(len(self._data)))
        return fp

    def register(self, out_fname, info):
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
        """
        Flush out a line

        Parameters
        ----------
        line

        Returns
        -------

        """
        self._fp.write("{}\n".format(json.dumps(line)))

    def dump(self):
        """
        Rewrite the entire file
        """
        logger.info("Rewriting output file")
        self._fp.close()
        # Create a new file truncating the old values
        self._fp = open(join(self.out_dir, self.filename), "w")
        for _, l in self._data.items():
            self._write_line(l)
        self._fp.flush()
