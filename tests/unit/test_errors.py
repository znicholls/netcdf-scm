import warnings


from netcdf_scm.errors import raise_no_iris_warning


def test_raise_no_iris_warning():
    expected_warn = (
        "A compatible version of Iris is not installed, not all functionality will "
        "work. We recommend installing the lastest version of Iris using conda to "
        "address this."
    )

    with warnings.catch_warnings(record=True) as no_iris_warning:
        raise_no_iris_warning()

    assert len(no_iris_warning) == 1
    assert str(no_iris_warning[0].message) == expected_warn
