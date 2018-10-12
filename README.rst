.. image:: https://raw.githubusercontent.com/znicholls/netcdf-scm/master/docs/source/_static/logo.png
   :height: 100
   :width: 200
   :scale: 50
   :alt: Logo


NetCDF SCM
==========

**Warning: `netcdf_scm` is still under heavy development so things are likely to change fast**

+----------------+-----------+--------+--------+-------------------+--------------+
| |Build Status| | |Codecov| | |Docs| | |PyPI| | |Python Versions| | |JOSS paper| |
+----------------+-----------+--------+--------+-------------------+--------------+

+-------------+------------------+---------------+------------------------------+----------------+----------+-----------+
| |Downloads| | |Latest Version| | |Last Commit| | |Commits Since Last Release| | |Contributors| | |Zenodo| | |License| |
+-------------+------------------+---------------+------------------------------+----------------+----------+-----------+

.. sec-begin-index

NetCDF SCM is a package of Python wrappers for processing netCDF files for use with simple climate models, built on top of the Iris_ package.

.. _Iris: https://github.com/SciTools/iris

.. sec-end-index

Documentation
-------------

Documentation can be found at [add link here]. For our docstrings we use numpy style docstrings.
For more information on these, `here is the full guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and `the quick reference we also use <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

For documentation we use `Sphinx <http://www.sphinx-doc.org/en/master/>`_.
To get ourselves started with Sphinx, we started with `this example <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_ then used `Sphinx's getting started guide <http://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.

Gotchas
~~~~~~~

To get this to work completely, you require `Latexmk <https://mg.readthedocs.io/latexmk.html>`_.
On a Mac this can be installed with ``sudo tlmgr install latexmk``.
You will most likely also need to install some other packages (if you don't have the full distribution).
You can check which package contains any missing files with ``tlmgr search --global --file [filename]``.
You can then install the packages with ``sudo tlmgr install [package]``.


Why is there a `Makefile` in a pure Python repository?
------------------------------------------------------

Whilst it's probably not good practice, a ``Makefile`` is a simple way to automate general setup (environment setup in particular).
Hence we have one here which basically acts as a notes file for how to do all those little jobs which we often forget e.g. setting up environments, running tests (and making sure we're in the right environment), building docs, setting up auxillary bits and pieces.


Why did we choose a BSD 2-Clause License?
-----------------------------------------

We want to ensure that our code can be used and shared as easily as possible.
Whilst we love transparency, we didn't want to **force** all future users to also comply with a stronger license such as AGPL.
Hence the choice we made.

.. |Build Status| image:: https://travis-ci.org/znicholls/netcdf-scm.svg?branch=master
    :target: https://travis-ci.org/znicholls/netcdf-scm
.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
    :target: https://netcdf-scm.readthedocs.io/en/latest/
.. |Codecov| image:: https://img.shields.io/codecov/c/github/znicholls/netcdf-scm.svg
    :target: https://codecov.io/gh/znicholls/netcdf-scm
.. |PyPI| image:: https://img.shields.io/pypi/v/netcdf-scm.svg
    :target: https://pypi.org/project/netcdf-scm/
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/netcdf-scm.svg
    :target: https://pypi.org/project/netcdf-scm/
.. |JOSS Paper| image:: https://joss.theoj.org/papers/paper-code/status.svg
    :target: https://joss.theoj.org/papers/paper-code
.. |Downloads| image:: https://img.shields.io/conda/dn/conda-forge/netcdf-scm.svg
    :target: https://anaconda.org/conda-forge/netcdf-scm
.. |Latest Version| image:: https://img.shields.io/github/tag/znicholls/netcdf-scm.svg
    :target: https://github.com/znicholls/netcdf-scm/releases
.. |Last Commit| image:: https://img.shields.io/github/last-commit/znicholls/netcdf-scm.svg
    :target: https://github.com/znicholls/netcdf-scm/commits/master
.. |Commits Since Last Release| image:: https://img.shields.io/github/commits-since/znicholls/netcdf-scm/latest.svg
    :target: https://github.com/znicholls/netcdf-scm/commits/master
.. |Contributors| image:: https://img.shields.io/github/contributors/znicholls/netcdf-scm.svg
    :target: https://github.com/znicholls/netcdf-scm/graphs/contributors
.. |Zenodo| image:: https://zenodo.org/badge/doi-no.svg
    :target: https://zenodo.org/badge/latestdoi/doi-no
.. |License| image:: https://img.shields.io/github/license/znicholls/netcdf-scm.svg
    :target: https://github.com/znicholls/netcdf-scm/blob/master/LICENSE

