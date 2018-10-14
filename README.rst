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

License
-------

.. sec-begin-license

NetCDF-SCM is free software under a BSD 2-Clause License, see `LICENSE <./LICENSE>`_.
(When it's written), if you make any use of NetCDF-SCM, please cite `The Journal of Open Source Software (JOSS) <http://joss.theoj.org/>`_ paper [insert reference here]:

.. sec-end-license

.. sec-begin-installation

Installation
------------

[To be written and tested after merge. Gist is can install with pip but iris dependencies are hard so better to use conda].

The easiest way to install NetCDF-SCM is with `conda <https://conda.io/miniconda.html>`_

::

    conda install -c conda-forge netcdf-scm

It is also possible to install it with `pip <https://pypi.org/project/pip/>`_

::

  pip install netcdf-scm

However installing with pip requires installing all of Iris_'s dependencies yourself which is not trivial.

.. _Iris: https://github.com/SciTools/iris

.. sec-end-installation

Documentation
-------------

Documentation can be found at `Read the Docs <https://netcdf-scm.readthedocs.io/en/latest/>`_.


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

.. [Morin et al. 2012]: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002598
