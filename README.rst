.. image:: https://raw.githubusercontent.com/znicholls/netcdf-scm/master/docs/source/_static/logo.png
   :height: 100
   :width: 200
   :scale: 50
   :alt: Logo


NetCDF SCM
==========

+--------+-------------------+-------------+-----------+--------+-----------------+
| Basics | |Python Versions| | |Platforms| | |License| | |Docs| | |Conda install| |
+--------+-------------------+-------------+-----------+--------+-----------------+

+-----------+--------------+----------+
| Citations | |JOSS paper| | |Zenodo| |
+-----------+--------------+----------+

+-------------------+----------------+-----------+
| Repository health | |Build Status| | |Codecov| |
+-------------------+----------------+-----------+

+-----------------+------------+--------+------------------+-------------+
| Latest releases | |Anaconda| | |PyPI| | |Latest Version| | |Downloads| |
+-----------------+------------+--------+------------------+-------------+

+-----------------+----------------+---------------+------------------------------+
| Latest activity | |Contributors| | |Last Commit| | |Commits Since Last Release| |
+-----------------+----------------+---------------+------------------------------+

.. sec-begin-index

NetCDF SCM is a Python package for processing netCDF files.
It focusses on metrics which are relevant to simple climate models and is built on top of the Iris_ package.

.. _Iris: https://github.com/SciTools/iris

.. sec-end-index

License
-------

.. sec-begin-license

NetCDF-SCM is free software under a BSD 2-Clause License, see `LICENSE <./LICENSE>`_.
If you make any use of NetCDF-SCM, please cite `The Journal of Open Source Software (JOSS) <http://joss.theoj.org/>`_ paper [insert reference here when written...] as well as the relevant `Zenodo release <https://zenodo.org/search?page=1&size=20&q=netcdf-scm>`_.

.. sec-end-license

.. sec-begin-installation

Installation
------------

The easiest way to install NetCDF-SCM is with `conda <https://conda.io/miniconda.html>`_

::

    # if you're using a conda environment, make sure you're in it
    conda install -c conda-forge netcdf-scm

If you do install it this way, we think (but aren't yet completely sure) that you will also need to install (at least) the minimal pip requirements.

::

  # if you're using a conda environment, make sure you're in it
  # and that pip is installed in the conda environment
  pip install -Ur pip-requirements-minimal.txt

It is also possible to install it with `pip <https://pypi.org/project/pip/>`_

::

  # if you're using a virtual environment, make sure you're in it
  pip install netcdf-scm

However installing with pip requires installing all of Iris_'s dependencies yourself which is not trivial.

.. _Iris: https://github.com/SciTools/iris

.. sec-end-installation

Documentation
-------------

Documentation can be found at our `documentation pages <https://netcdf-scm.readthedocs.io/en/latest/>`_ (we are thankful to `Read the Docs <https://readthedocs.org/>`_ for hosting us).


Contributing
------------

Please see the `Development section of the docs <https://netcdf-scm.readthedocs.io/en/latest/development.html>`_.

.. |Build Status| image:: https://travis-ci.com/znicholls/netcdf-scm.svg?branch=master
    :target: https://travis-ci.com/znicholls/netcdf-scm
.. |Docs| image:: https://readthedocs.org/projects/netcdf-scm/badge/?version=latest
    :target: https://netcdf-scm.readthedocs.io/en/latest/
.. |Codecov| image:: https://img.shields.io/codecov/c/github/znicholls/netcdf-scm.svg
    :target: https://codecov.io/gh/znicholls/netcdf-scm
.. |PyPI| image:: https://img.shields.io/pypi/v/netcdf-scm.svg
    :target: https://pypi.org/project/netcdf-scm/
.. |Anaconda| image:: https://anaconda.org/conda-forge/netcdf-scm/badges/version.svg
    :target: https://anaconda.org/conda-forge/netcdf-scm
.. |Platforms| image:: https://anaconda.org/conda-forge/netcdf-scm/badges/platforms.svg
    :target: https://anaconda.org/conda-forge/netcdf-scm
.. |Conda install| image:: https://anaconda.org/conda-forge/netcdf-scm/badges/installer/conda.svg
    :target: https://conda.anaconda.org/conda-forge
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
.. |Zenodo| image:: https://zenodo.org/badge/151593566.svg
    :target: https://zenodo.org/badge/latestdoi/151593566
.. |License| image:: https://img.shields.io/github/license/znicholls/netcdf-scm.svg
    :target: https://github.com/znicholls/netcdf-scm/blob/master/LICENSE

.. [Morin et al. 2012]: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002598
