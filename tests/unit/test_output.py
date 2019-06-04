import re

import json
import pytest
from os.path import exists, join
from unittest.mock import Mock

from netcdf_scm.output import OutputFileDatabase


def test_empty_tracker(tmpdir):
    tracker = OutputFileDatabase(tmpdir)

    tracker_fname = join(tmpdir, tracker.filename)
    assert exists(tracker_fname)
    assert open(tracker_fname).read() == ""


def test_existing_tracker(tmpdir):
    tracker_fname = join(tmpdir, "netcdf-scm_crunched.jsonl")
    exp = {"filename": "test"}
    with open(tracker_fname, "w") as fh:
        fh.write("{}\n".format(json.dumps(exp)))

    tracker = OutputFileDatabase(tmpdir)
    assert len(tracker) == 1

    assert tracker._data["test"] == exp


def test_existing_tracker_corrupted(tmpdir):
    tracker_fname = join(tmpdir, "netcdf-scm_crunched.jsonl")
    with open(tracker_fname, "w") as fh:
        fh.write("{}\n".format(json.dumps({"filename": "test"})))
        fh.write("{}\n".format(json.dumps({"filename": "test"})))

    error_msg = re.escape("Corrupted output file: duplicate entries for filename")
    with pytest.raises(ValueError):
        OutputFileDatabase(tmpdir)


def test_close_and_open(tmpdir):
    exp = {"filename": "test", "metadata": ["other"]}
    exp2 = {"filename": "test2", "metadata": ["other2"]}

    tracker = OutputFileDatabase(tmpdir)
    assert len(tracker) == 0
    tracker.register("test", exp)
    tracker.register("test2", exp2)
    assert len(tracker) == 2

    tracker2 = OutputFileDatabase(tmpdir)
    assert len(tracker2) == 2
    assert tracker2._data["test"]["metadata"] == exp["metadata"]
    assert tracker2._data["test2"]["metadata"] == exp2["metadata"]


def test_overwriting(tmpdir):
    orig = {"filename": "test", "metadata": ["other"]}
    exp = {"filename": "test", "metadata": ["other_overwritten"]}

    tracker = OutputFileDatabase(tmpdir)
    tracker.register("test", orig)
    tracker.register("line1", {"filename": "line1", "metadata": ["testing"]})
    assert list(tracker._data.keys()) == ["test", "line1"]

    tracker.dump = Mock(side_effect=tracker.dump)

    tracker.register("test", exp)
    tracker.dump.assert_called()
    assert list(tracker._data.keys()) == ["line1", "test"]

    tracker2 = OutputFileDatabase(tmpdir)
    assert tracker2._data["test"]["metadata"] == exp["metadata"]
    assert tracker2._data["line1"]["metadata"] == ["testing"]
    assert list(tracker2._data.keys()) == ["line1", "test"]
