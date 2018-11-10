import warnings


def raise_no_iris_warning():
    warn_msg = (
        "A compatible version of Iris is not installed, not all functionality will "
        "work. We recommend installing the lastest version of Iris using conda to "
        "address this."
    )
    warnings.warn(warn_msg)
