"""
Tests for ugrid_checks.nc_scan_dataset.scan_dataset

"""

import numpy as np
from numpy import asanyarray as nparray
from pytest import fixture
from ugrid_checks.nc_dataset_scan import NcDimSummary
from ugrid_checks.tests import cdl_scanner

# Prevent error from 'black' about unused import.
# NOTE : the import is *not* in fact redundant, since pytest requires it.
cdl_scanner


@fixture()
def standard_dataset_testscan(cdl_scanner):
    standard_dataset_test_cdl = """
        netcdf small_test {
        dimensions:
            x = 3 ;
            y = 4 ;
            t = UNLIMITED ;

        variables:
            double x(x) ;
                x:name = "x-coord" ;
            int y(y) ;
                y:name = "y-coord" ;
            float t(t) ;
                t:name = "time" ;
                t:units = "s" ;
            float aux_x(x) ;
                aux_x:units = "1" ;
            float data_var(t, y, x) ;
                data_var:name = "multidim_data" ;
                data_var:units = "1" ;
                data_var:_FillValue = 9.999e25f ;
                data_var:coordinates = "aux_x" ;

        // global attributes:
            :Conventions = "CF-1.0" ;
            :comment = "global comment attribute" ;

        data:
            t = 1.0, 2.0 ;
        }
    """
    return cdl_scanner.scan(standard_dataset_test_cdl)


def test_dimensions(standard_dataset_testscan):
    dims = standard_dataset_testscan.dimensions
    assert len(dims) == 3
    assert list(dims.keys()) == ["x", "y", "t"]
    assert dims["x"] == NcDimSummary(name="x", length=3, is_unlimited=False)
    assert dims["y"] == NcDimSummary(name="y", length=4, is_unlimited=False)
    assert dims["t"] == NcDimSummary(name="t", length=2, is_unlimited=True)


def test_variables(standard_dataset_testscan):
    vars = standard_dataset_testscan.variables
    assert len(vars) == 5
    assert set(vars.keys()) == set(["x", "y", "t", "aux_x", "data_var"])

    # Note: testing with individual asserts ... it is _possible_ to check
    # for equality with an expected, matching NcVariableSummary, but this
    # makes any failure messages very hard to interpret.
    var_x = vars["x"]
    assert var_x.name == "x"
    assert var_x.dimensions == ("x",)
    assert var_x.shape == (3,)
    assert var_x.dtype == np.float64
    assert var_x.attributes == {"name": "x-coord"}

    var_y = vars["y"]
    assert var_y.name == "y"
    assert var_y.dimensions == ("y",)
    assert var_y.shape == (4,)
    assert var_y.dtype == np.int32
    assert var_y.attributes == {"name": "y-coord"}

    var_t = vars["t"]
    assert var_t.name == "t"
    assert var_t.dimensions == ("t",)
    assert var_t.shape == (2,)
    assert var_t.dtype == np.float32
    assert var_t.attributes == {"name": "time", "units": "s"}

    var_auxx = vars["aux_x"]
    assert var_auxx.name == "aux_x"
    assert var_auxx.dimensions == ("x",)
    assert var_auxx.shape == (3,)
    assert var_auxx.dtype == np.float32
    assert var_auxx.attributes == {"units": "1"}

    var_data = vars["data_var"]
    assert var_data.name == "data_var"
    assert var_data.dimensions == ("t", "y", "x")
    assert var_data.shape == (2, 4, 3)
    assert var_data.dtype == np.float32
    assert var_data.attributes == {
        "units": nparray("1"),
        "name": nparray("multidim_data"),
        "_FillValue": nparray(9.999e25, dtype=np.float32),
        "coordinates": nparray("aux_x"),
    }


def test_global_attributes(standard_dataset_testscan):
    attrs = standard_dataset_testscan.attributes
    assert len(attrs) == 2
    assert list(attrs.keys()) == ["Conventions", "comment"]
    assert attrs["Conventions"] == nparray("CF-1.0")
    assert attrs["comment"] == nparray("global comment attribute")


def test_variable_content(standard_dataset_testscan):
    from ..._var_data import VariableDataProxy

    var_t = standard_dataset_testscan.variables["t"]
    data = var_t.data
    assert isinstance(data, VariableDataProxy)
    array = data.fetch_array()
    assert isinstance(array, np.ndarray)
    assert array.dtype == var_t.dtype
    assert array.shape == var_t.shape
    assert np.all(array == [1.0, 2.0])
