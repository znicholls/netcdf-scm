from unittest.mock import patch
import warnings
from importlib import reload
import builtins


__REALIMPORT__ = builtins.__import__


def mock_iris_import_error(name, globals, locals, fromlist, level):
    if name == "iris":
        raise ModuleNotFoundError
    return __REALIMPORT__(name, globals, locals, fromlist, level)


def test_iris_cube_wrappers_no_iris_warning():
    with warnings.catch_warnings(record=True) as no_iris_warnings:
        with patch("builtins.__import__", side_effect=mock_iris_import_error):
            import netcdf_scm.iris_cube_wrappers

            reload(netcdf_scm.iris_cube_wrappers)

    expected_warn = (
        "A compatible version of Iris is not installed, not all functionality will "
        "work. We recommend installing the lastest version of Iris using conda to "
        "address this."
    )
    assert len(no_iris_warnings) == 1
    assert str(no_iris_warnings[0].message) == expected_warn

    # put everything back
    with warnings.catch_warnings(record=True) as with_iris_warnings:
        reload(netcdf_scm.iris_cube_wrappers)

    assert len(with_iris_warnings) == 0

def test_utils_no_iris_warning():
    with warnings.catch_warnings(record=True) as no_iris_warnings:
        with patch("builtins.__import__", side_effect=mock_iris_import_error):
            import netcdf_scm.utils

            reload(netcdf_scm.utils)

    expected_warn = (
        "A compatible version of Iris is not installed, not all functionality will "
        "work. We recommend installing the lastest version of Iris using conda to "
        "address this."
    )
    assert len(no_iris_warnings) == 1
    assert str(no_iris_warnings[0].message) == expected_warn

    # put everything back
    with warnings.catch_warnings(record=True) as with_iris_warnings:
        reload(netcdf_scm.utils)

    assert len(with_iris_warnings) == 0
