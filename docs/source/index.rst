.. netcdf-scm documentation master file, created by
   sphinx-quickstart on Fri Oct 12 23:11:20 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NetCDF-SCM
==========

.. include:: ../../README.rst
    :start-after: sec-begin-index
    :end-before: sec-end-index

.. toctree::
   :maxdepth: 2
   :caption: Documentation


.. toctree::
    :maxdepth: 2
    :caption: API reference

    utils
    iris-cube-wrappers


A note on masking
-----------------

The convention is that a value of ``True`` in a mask indicates that the value is invalid.
Hence a ``land_mask`` should be ``False`` where the associated data is land and ``True`` where the associated data is ocean.
This can be confusing as then a land and northern hemisphere mask have to be combined with 'or' logic rather than 'and' logic i.e. ``land_nh_mask = land_mask or nh_mask``, not ``land_nh_mask = land_mask and nh_mask`` as one might intuitively expect.
However, this is the convention so we should follow it.
(A way that may to think about it is that we want to mask where we're not on land or where we're not in the northern hemisphere, leaving only the regions where we're both on land and in the northern hemisphere).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
