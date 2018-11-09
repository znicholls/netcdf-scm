Usage
=====

.. contents:: Contents
    :local:

All of our usage examples are included in ``netcdf-scm/notebooks``.
Command line usage examples can be found in the help message of our command line interface (which we're still building, see `this issue <https://github.com/znicholls/netcdf-scm/issues/6>`_).


Masking
-------

In a masked array, the convention is that a value of ``True`` in a mask indicates that the value is invalid.
Hence a ``land_mask`` should be ``False`` where the associated data is land and ``True`` where the associated data is ocean.
This can be confusing as then a land and northern hemisphere mask have to be combined with 'or' logic rather than 'and' logic i.e. ``land_nh_mask = land_mask or nh_mask``, not ``land_nh_mask = land_mask and nh_mask`` as one might intuitively expect.
(One way to think about it is that we want to mask where we're not on land or where we're not in the northern hemisphere, leaving only the regions where we're both on land and in the northern hemisphere).
