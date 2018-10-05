### A note on masking

The convention is that a value of True in a mask indicates that the value is invalid.
Hence a `land_mask` should have False where it's land and True where it's ocean.
This can be confusing as then a land and northern hemisphere mask have to be combined with 'or' logic rather than 'and' logic i.e. `land_nh_mask = land_mask or nh_mask`.
However, this is the convention so we should follow it (a way that may help users to think about it is that we want to mask where we're not on land or where we're not in the NH, leaving only the regions where we're both on land and in the NH).
