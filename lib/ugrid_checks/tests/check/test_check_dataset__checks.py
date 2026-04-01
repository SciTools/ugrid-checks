"""
Tests for ugrid_checks.check.check_dataset

"""

import logging
import re

import numpy as np
from pytest import fixture
from ugrid_checks._var_data import VariableDataProxy
from ugrid_checks.check import check_dataset
from ugrid_checks.nc_dataset_scan import NcDimSummary, NcVariableSummary

from .. import cdl_scanner, next_mesh, next_var

# Prevent error from 'black' about unused import.
# NOTE : the import is *not* in fact redundant, since pytest requires it.
cdl_scanner


@fixture
def scan_2d_mesh(cdl_scanner):
    """
    Return a scan representing a 'typical' testfile with a mesh.

    Being re-created for every test, this can be post-modified to exercise the
    individual conformance tests.
    (which is easier than making modified actual files or CDL strings)

    """
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        face_dim = 6 ;
        num_node = 8 ;
        num_vertices = 4 ;

    variables:
        double sample_data(face_dim) ;
            sample_data:long_name = "sample_data" ;
            sample_data:coordinates = "latitude longitude" ;
            sample_data:location = "face" ;
            sample_data:mesh = "topology" ;
        double latitude(face_dim) ;
            latitude:units = "degrees_north" ;
            latitude:standard_name = "latitude" ;
            latitude:long_name = "latitude of 2D face centres" ;
            latitude:bounds_long_name = "latitude of 2D mesh faces." ;
        double longitude(face_dim) ;
            longitude:units = "degrees_east" ;
            longitude:standard_name = "longitude" ;
            longitude:long_name = "longitude of 2D face centres" ;
            longitude:bounds_long_name = "longitude of 2D mesh faces." ;
        double node_lat(num_node) ;
            node_lat:standard_name = "latitude" ;
            node_lat:long_name = "latitude of 2D mesh nodes." ;
            node_lat:units = "degrees_north" ;
        double node_lon(num_node) ;
            node_lon:standard_name = "longitude" ;
            node_lon:long_name = "longitude of 2D mesh nodes." ;
            node_lon:units = "degrees_east" ;
        int face_nodes(face_dim, num_vertices) ;
            face_nodes:long_name = "Map every face to its corner nodes." ;
            face_nodes:cf_role = "face_node_connectivity" ;
            face_nodes:start_index = 1 ;
        int topology ;
            topology:cf_role = "mesh_topology" ;
            topology:topology_dimension = 2L ;
            topology:node_coordinates = "node_lat node_lon" ;
            topology:face_coordinates = "latitude longitude" ;
            topology:face_node_connectivity = "face_nodes" ;
            topology:face_dimension = "face_dim" ;
            topology:long_name = "Topology data of 2D unstructured mesh" ;

    // global attributes:
            :Conventions = "UGRID-1.0" ;
    }
    """
    return cdl_scanner.scan(test_cdl)


@fixture
def scan_mini_w_data(cdl_scanner):
    """
    A scan for a tiny 2d mesh with all data values, for data-specific checks.

    """
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        faces = 2 ;
        nodes = 5 ;
        face_n_vertices = 4 ;

    variables:
        double sample_data(faces) ;
            sample_data:long_name = "sample_data" ;
            sample_data:coordinates = "x y" ;
            sample_data:location = "face" ;
            sample_data:mesh = "topology" ;
        double node_y(nodes) ;
            node_y:standard_name = "latitude" ;
            node_y:units = "degrees_north" ;
        double node_x(nodes) ;
            node_x:standard_name = "longitude" ;
            node_x:units = "degrees_east" ;
        int face_nodes(faces, face_n_vertices) ;
            face_nodes:cf_role = "face_node_connectivity" ;
            face_nodes:start_index = 1 ;
            face_nodes:_FillValue = -1 ;
        int topology ;
            topology:cf_role = "mesh_topology" ;
            topology:topology_dimension = 2L ;
            topology:node_coordinates = "node_x node_y" ;
            topology:face_node_connectivity = "face_nodes" ;

    // global attributes:
            :Conventions = "UGRID-1.0" ;

    data:
        node_x = 0., 0., 1., 1., 2. ;
        node_y = 0., -1., -1., 0., -1. ;
        face_nodes = 1, 2, 3, 4, 4, 3, 5, -1 ;
    }
    """
    # The above represents a ~minimal flexible 2d mesh,
    # like this ..
    # 1 - 4
    # |   | \
    # 2 - 3 - 5
    return cdl_scanner.scan(test_cdl)


@fixture
def scan_lis_w_data(cdl_scanner):
    """
    A scan for a small 2d mesh including an LIS.

    """
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        faces = 3 ;
        nodes = 6 ;
        lis_faces = 2 ;
        face_n_vertices = 4 ;

    variables:
        double sample_data(lis_faces) ;
            sample_data:long_name = "sample_data" ;
            sample_data:coordinates = "x y" ;
            sample_data:location_index_set = "lis" ;
        int lis(lis_faces) ;
            lis:cf_role = "location_index_set" ;
            lis:mesh = "topology" ;
            lis:location = "face" ;
            lis:start_index = 1 ;
        double node_y(nodes) ;
            node_y:standard_name = "latitude" ;
            node_y:units = "degrees_north" ;
        double node_x(nodes) ;
            node_x:standard_name = "longitude" ;
            node_x:units = "degrees_east" ;
        int face_nodes(faces, face_n_vertices) ;
            face_nodes:cf_role = "face_node_connectivity" ;
            face_nodes:start_index = 1 ;
            face_nodes:_FillValue = -1 ;
        int topology ;
            topology:cf_role = "mesh_topology" ;
            topology:topology_dimension = 2L ;
            topology:node_coordinates = "node_x node_y" ;
            topology:face_node_connectivity = "face_nodes" ;

    // global attributes:
            :Conventions = "UGRID-1.0" ;

    data:
        node_x = 100., 101., 102., 103., 104., 105. ;
        node_y = 200., 201., 202., 203., 204., 205. ;
        face_nodes = 1, 2, 3, 4,  4, 3, 5, 6,  2, 5, 6, 1 ;
        lis = 2, 1 ;
    }
    """
    return cdl_scanner.scan(test_cdl)


@fixture
def scan_1d_mesh(cdl_scanner):
    """Return a scan representing a 1d (edges only) mesh."""
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        edge_dim = 6 ;
        num_node = 8 ;
        num_ends = 2 ;

    variables:
        double sample_data(edge_dim) ;
            sample_data:long_name = "sample_data" ;
            sample_data:coordinates = "latitude longitude" ;
            sample_data:location = "edge" ;
            sample_data:mesh = "topology" ;
        double latitude(edge_dim) ;
            latitude:units = "degrees_north" ;
            latitude:standard_name = "latitude" ;
            latitude:long_name = "latitude of 2D edge centres" ;
            latitude:bounds_long_name = "latitude of 2D mesh edges." ;
        double longitude(edge_dim) ;
            longitude:units = "degrees_east" ;
            longitude:standard_name = "longitude" ;
            longitude:long_name = "longitude of 2D edge centres" ;
            longitude:bounds_long_name = "longitude of 2D mesh edges." ;
        double node_lat(num_node) ;
            node_lat:standard_name = "latitude" ;
            node_lat:long_name = "latitude of 2D mesh nodes." ;
            node_lat:units = "degrees_north" ;
        double node_lon(num_node) ;
            node_lon:standard_name = "longitude" ;
            node_lon:long_name = "longitude of 2D mesh nodes." ;
            node_lon:units = "degrees_east" ;
        int edge_nodes(edge_dim, num_ends) ;
            edge_nodes:long_name = "Map every edge to its endpoints." ;
            edge_nodes:cf_role = "edge_node_connectivity" ;
            edge_nodes:start_index = 1 ;
        int topology ;
            topology:cf_role = "mesh_topology" ;
            topology:topology_dimension = 1L ;
            topology:node_coordinates = "node_lat node_lon" ;
            topology:edge_coordinates = "latitude longitude" ;
            topology:edge_node_connectivity = "edge_nodes" ;
            topology:edge_dimension = "edge_dim" ;
            topology:long_name = "Topology data of 1D unstructured mesh" ;

    // global attributes:
            :Conventions = "UGRID-1.0" ;
    }
    """
    return cdl_scanner.scan(test_cdl)


@fixture
def scan_0d_mesh(cdl_scanner):
    """Return a scan representing a 0d (nodes only) mesh."""
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        num_node = 8 ;

    variables:
        double sample_data(num_node) ;
            sample_data:long_name = "sample_data" ;
            sample_data:location = "node" ;
            sample_data:mesh = "topology" ;
        double node_lat(num_node) ;
            node_lat:standard_name = "latitude" ;
            node_lat:long_name = "latitude of 2D mesh nodes." ;
            node_lat:units = "degrees_north" ;
        double node_lon(num_node) ;
            node_lon:standard_name = "longitude" ;
            node_lon:long_name = "longitude of 2D mesh nodes." ;
            node_lon:units = "degrees_east" ;
        int topology ;
            topology:cf_role = "mesh_topology" ;
            topology:topology_dimension = 0L ;
            topology:node_coordinates = "node_lat node_lon" ;
            topology:long_name = "Topology data of 0D unstructured mesh" ;

    // global attributes:
            :Conventions = "UGRID-1.0" ;
    }
    """
    return cdl_scanner.scan(test_cdl)


class FakeVariableDataWrapper(VariableDataProxy):
    def __init__(self, original_proxy, array):
        self._original_proxy = original_proxy
        self.array = np.asanyarray(array)

    def fetch_array(self):
        return self.array

    @property
    def datapath(self):
        # Mirror the wrapped content : fail if there was none
        return self._original_proxy.datapath

    @property
    def varname(self):
        # Mirror the wrapped content : fail if there was none
        return self._original_proxy.varname


class DatasetChecker:
    # Generic helper functions for dataset-scan testing.
    def _check_dataset(self, scan, *args, **kwargs):
        # Conformance-check the given scan.
        checker = check_dataset(scan, print_summary=False, *args, **kwargs)
        logs = checker.logger.report_statement_logrecords()
        return logs

    @staticmethod
    def _expect_notes(statements, expected_notes):
        # Test that each entry in 'expected' matches one of 'statements',
        # and that all 'statements' were matched.
        def note_from_logrecord(record):
            statement_code = None
            if record.levelname == "INFO":
                statement_code = ""
            elif record.levelname == "WARNING":
                statement_code = "A" + str(record.args[0])
            elif record.levelname == "ERROR":
                statement_code = "R" + str(record.args[0])
            assert statement_code is not None
            return (statement_code, record.msg)

        actual_notes = [
            note_from_logrecord(record)
            for record in statements
            if record.levelno >= logging.INFO
        ]
        # Replace each expected item with a matching actual one,
        # if one can be found.
        for i_expected, expected_note in enumerate(expected_notes.copy()):
            expected_code, expected_msg = expected_note
            for actual_note in actual_notes:
                actual_code, actual_msg = actual_note
                if (
                    expected_code is None or expected_code == actual_code
                ) and re.search(expected_msg, actual_msg):
                    expected_notes[i_expected] = actual_note
                    break  # Only ever take first match
        # This should result in a matching list.
        assert set(actual_notes) == set(expected_notes)

    def check(
        self,
        scan,
        code=None,
        message=None,
        statements=None,
        with_datachecks=False,
    ):
        # Check statements generated by checking a dataset scan.
        # Usage forms :
        #   check(scan)  : expect no statements
        #   check(scan, code, message)  : expect exactly 1 statement
        #   check(scan, statements=[(code1, message1), ...])  : multiples
        datacheck_size = -1.0 if with_datachecks else 0.0
        logs = self._check_dataset(scan, max_data_mb=datacheck_size)
        if statements is None:
            if code is None and message is None:
                # Expect *no* statements.
                statements = []
            else:
                # Expect code+message to specify a single problem.
                statements = [(code, message)]
        self._expect_notes(logs, statements)

    def check_withdata(self, *args, **kwargs):
        kwargs["with_datachecks"] = True
        self.check(*args, **kwargs)

    @classmethod
    def _add_edges(cls, scan, n_edges=5, with_facedge_conn=False):
        # Add edges and an edge-node connectivity to a 2d scan.
        meshvar = scan.variables["topology"]
        # Check we have the 2d scan, as 1d defines a different edges dim
        assert meshvar.attributes["topology_dimension"] == 2
        scan.dimensions["edges"] = NcDimSummary("edges", n_edges)
        scan.dimensions["n_edge_ends"] = NcDimSummary("n_edge_ends", 2)
        edgenodes_name = "edge_nodes"
        scan.variables[edgenodes_name] = NcVariableSummary(
            name=edgenodes_name,
            dimensions=["edges", "n_edge_ends"],
            shape=(n_edges, 2),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "edge_node_connectivity"},
        )
        meshvar.attributes["edge_node_connectivity"] = edgenodes_name

        if with_facedge_conn:
            cls._add_faceedge_conn(scan)

    @staticmethod
    def _add_faceedge_conn(scan):
        """
        Add a face-edgenode (optional) connectivity to a 2d scan.

        Mesh needs to have suitable varnames :  should be produced by calling
        'self._add_edges' first.

        """
        # Now add the (optional) face-edge connectivity.
        scan.dimensions["face_n_edges"] = NcDimSummary("face_n_edges", 4)
        faceedge_name = "face_edges_var"
        edgeface_conn = NcVariableSummary(
            name=faceedge_name,
            dimensions=["face_n_edges", "num_vertices"],
            shape=(5, 4),
            dtype=np.dtype(np.int64),
            attributes={"cf_role": "face_edge_connectivity"},
        )
        scan.variables[faceedge_name] = edgeface_conn
        meshvar = scan.variables["topology"]
        meshvar.attributes["face_edge_connectivity"] = faceedge_name

    @staticmethod
    def _convert_to_lis(scan):
        # Convert a scan so the data is mapped via a location-index-set
        # Should work on any of the basic 0d/1d/2d scans.
        # Add a new location-index-set dimension, and variable
        location = scan.variables["sample_data"].attributes["location"]
        lis_dim_name = f"lis_{location}s"
        scan.dimensions[lis_dim_name] = NcDimSummary(lis_dim_name, 3)
        lis_name = "lis"
        scan.variables[lis_name] = NcVariableSummary(
            name=lis_name,
            dimensions=(lis_dim_name,),
            shape=(5, 3),
            dtype=np.dtype(np.int64),
            attributes={
                "cf_role": "location_index_set",
                "mesh": "topology",
                "location": location,
            },
        )

        # 'Switch' the data var to reference the lis
        data_var = scan.variables["sample_data"]
        data_var.dimensions = (lis_dim_name,)
        data_var.shape = (3,)
        del data_var.attributes["mesh"]
        del data_var.attributes["location"]
        data_var.attributes["location_index_set"] = lis_name


class TestChecker_Dataset(DatasetChecker):
    # Test dataset-level checking.
    def test_mesh_0d_noerror(self, scan_0d_mesh):
        # Check that "0d" example, unmodified, is all OK
        self.check(scan_0d_mesh)

    def test_mesh_1d_noerror(self, scan_1d_mesh):
        # Check that "1d" example, unmodified is all OK
        self.check(scan_1d_mesh)

    def test_mesh_2d_noerror(self, scan_2d_mesh):
        # Check that "2d" example, unmodified is all OK
        self.check(scan_2d_mesh)

    def test_lis_0d_noerror(self, scan_0d_mesh):
        # Just check that we can process a simple LIS case successfully.
        self._convert_to_lis(scan_0d_mesh)
        data_attrs = scan_0d_mesh.variables["sample_data"].attributes
        assert "mesh" not in data_attrs
        assert "location_index_set" in data_attrs
        self.check(scan_0d_mesh)

    def test_1d_with_lis_noerror(self, scan_1d_mesh):
        # Check we can process a 1d LIS case (and the conversion utility works)
        self._convert_to_lis(scan_1d_mesh)
        data_attrs = scan_1d_mesh.variables["sample_data"].attributes
        assert "mesh" not in data_attrs
        assert "location_index_set" in data_attrs
        self.check(scan_1d_mesh)

    def test_2d_with_lis_noerror(self, scan_2d_mesh):
        # Check we can process a 2d LIS case (and the conversion utility works)
        self._convert_to_lis(scan_2d_mesh)
        data_attrs = scan_2d_mesh.variables["sample_data"].attributes
        assert "mesh" not in data_attrs
        assert "location_index_set" in data_attrs
        self.check(scan_2d_mesh)

    def test_minidata_noerror(self, scan_mini_w_data):
        self.check_withdata(scan_mini_w_data)

    def test_minidata_with_lis_noerror(self, scan_lis_w_data):
        self.check_withdata(scan_lis_w_data)

    def test_a104_dataset_shared_meshdims_2(self, scan_0d_mesh):
        # Copy 'topology' mesh and all its components to 'topology_2' etc
        next_mesh(scan_0d_mesh, "topology")
        # Fix the new node-coord variables to use the OLD nodes dim
        for xy in ("lon", "lat"):
            new_node_coord = scan_0d_mesh.variables[f"node_{xy}_2"]
            assert new_node_coord.dimensions == ["num_node_2"]
            new_node_coord.dimensions = ["num_node"]
        msg = (
            'Dimension "num_node" is mapped by both mesh "topology" and '
            'mesh "topology_2"'
        )
        self.check(scan_0d_mesh, "A104", msg)

    def test_a104_dataset_shared_meshdims_3(self, scan_0d_mesh):
        # Create 2 additional meshes
        next_mesh(scan_0d_mesh, "topology")  # makes 'topology_2'
        next_mesh(scan_0d_mesh, "topology_2")  # makes 'topology_3'
        # Fix the new node-coord variables to use the OLD nodes dim
        for imesh in (2, 3):
            for xy in ("lon", "lat"):
                new_node_coord = scan_0d_mesh.variables[f"node_{xy}_{imesh}"]
                assert new_node_coord.dimensions == [f"num_node_{imesh}"]
                new_node_coord.dimensions = ["num_node"]
        msg = (
            'Dimension "num_node" is mapped by multiple meshes : '
            '"topology", "topology_2" and "topology_3".'
        )
        self.check(scan_0d_mesh, "A104", msg)


class TestChecker_MeshVariables(DatasetChecker):
    # Simplified fixtures, when the meshvar is what is wanted.
    @fixture
    def scan_2d_and_meshvar(self, scan_2d_mesh):
        return scan_2d_mesh, scan_2d_mesh.variables["topology"]

    @fixture
    def scan_1d_and_meshvar(self, scan_1d_mesh):
        return scan_1d_mesh, scan_1d_mesh.variables["topology"]

    @fixture
    def scan_0d_and_meshvar(self, scan_0d_mesh):
        return scan_0d_mesh, scan_0d_mesh.variables["topology"]

    # Test mesh-variable checking.
    def test_r101_mesh_missing_cf_role(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        del meshvar.attributes["cf_role"]
        msg = r"no 'cf_role' property, which should be \"mesh_topology\"\."
        self.check(scan, "R101", msg)

    def test_r102_mesh_bad_cf_role(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["cf_role"] = "something odd"
        self.check(
            scan,
            statements=[
                ("R102", 'should be "mesh_topology"'),
                ("A905", "not a valid UGRID cf_role"),
            ],
        )

    def test_r103_mesh_no_topology_dimension(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        del meshvar.attributes["topology_dimension"]
        self.check(scan, "R103", r"no 'topology_dimension' attribute\.")

    def test_r104_mesh_unknown_topology_dimension(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["topology_dimension"] = 4
        self.check(scan, "R104", "not 0, 1 or 2")

    def test_r105_r107_mesh_badcoordattr_nonstring(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        # An invalid mesh-coord attribute.
        # This is always caused by a subsidiary specific error, so for testing
        # it doesn't much matter which we choose.
        meshvar.attributes["node_coordinates"] = 3
        msg = (
            "\"topology\" attribute 'node_coordinates'.*"
            "which is not a string type"
        )
        self.check(
            scan,
            statements=[
                ("R105", msg),
                ("R108", '"topology".*not a list of variables in the dataset'),
            ],
        )

    def test_r105_r108_mesh_badcoordattr_empty(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["node_coordinates"] = ""
        msg = (
            '"topology" has node_coordinates="".*'
            "is not a valid list of variable names"
        )
        self.check(
            scan,
            statements=[
                ("R105", msg),
                ("R108", '"topology".*not a list of variables'),
            ],
        )

    def test_r105_r108_mesh_badcoordattr_invalidname(
        self, scan_2d_and_meshvar
    ):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["node_coordinates"] = "$123"
        msg = (
            r'"topology" has node_coordinates="\$123"'
            ".*not a valid variable name"
        )
        self.check(
            scan,
            statements=[
                ("R105", msg),
                ("R108", '"topology".*not a list of variables'),
            ],
        )

    def test_r106_r107_mesh_badcoordattr_missingvar(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["node_coordinates"] = "unknown"
        msg = (
            r'\'node_coordinates\' refers to a variable "unknown", '
            r"but there is no such variable in the dataset\."
        )
        self.check(
            scan,
            statements=[
                ("R106", msg),
                ("R108", '"topology".*not a list of variables in the dataset'),
            ],
        )

    def test_r107_mesh_badconn_multiplevars(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        # Checking for multiple entries in a connectivity attribute.
        # Create extra custom variables, copying 'node_lat'
        next_var(scan, "node_lat")
        next_var(scan, "node_lat_2")
        # point a connectivity at those
        meshvar.attributes["face_node_connectivity"] = "node_lat_2 node_lat_3"
        # delete the now-unreferenced one, to avoid additional errors
        del scan.variables["face_nodes"]
        msg = (
            'face_node_connectivity="node_lat_2 node_lat_3"'
            ", which contains 2 names, instead of 1."
        )
        self.check(scan, "R107", msg)

    def test_r109_mesh_badconn_empty(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        # Checking the catchall error for an invalid mesh-coord attribute.
        # This is always caused by a subsidiary specific error, so for testing
        # it doesn't much matter which we choose.
        meshvar.attributes["face_node_connectivity"] = ""
        # delete the now-unreferenced one, to avoid additional errors
        del scan.variables["face_nodes"]
        self.check(
            scan,
            statements=[
                (
                    "R105",
                    (
                        '"topology" has face_node_connectivity="".*'
                        "not a valid list of variable names"
                    ),
                ),
                (
                    "R109",
                    (
                        '"topology" has face_node_connectivity=""'
                        ".*not a list of variables in the dataset"
                    ),
                ),
            ],
        )

    def test_r110_mesh_missing_node_coords(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        del meshvar.attributes["node_coordinates"]
        msg = "does not have a 'node_coordinates' attribute"
        self.check(scan, "R110", msg)

    def test_r111_mesh_topologydim0_extra_edgeconn(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        # Add extra dims and a variable, to mimic edge-node connectivity.
        # (which is not valid in a 0-d mesh)
        scan.dimensions["edges"] = NcDimSummary("edges", 3)
        scan.dimensions["edge_ends"] = NcDimSummary("edge_ends", 2)
        edgenodes_name = "fake_edge_nodes"
        scan.variables[edgenodes_name] = NcVariableSummary(
            name=edgenodes_name,
            dimensions=["edges", "edge_ends"],
            shape=(3, 2),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "edge_node_connectivity"},
        )
        meshvar.attributes["edge_node_connectivity"] = edgenodes_name
        msg = (
            'has topology_dimension="0", but the presence of.*'
            "'edge_node_connectivity'.*implies it should be 1."
        )
        self.check(scan, "R111", msg)

    def test_r112_mesh_topologydim1_missing_edgeconn(
        self, scan_1d_and_meshvar
    ):
        scan, meshvar = scan_1d_and_meshvar
        del meshvar.attributes["edge_node_connectivity"]
        # delete some other things, to avoid additional errors
        del scan.variables["edge_nodes"]
        del meshvar.attributes["edge_dimension"]
        # Also avoid checking the data-variable, which now has a dim problem.
        del scan.variables["sample_data"].attributes["mesh"]
        msg = 'has topology_dimension="1", but.*' "no 'edge_node_connectivity'"
        self.check(scan, "R112", msg)

    def test_r113_mesh_topologydim2_missing_faceconn(
        self, scan_1d_and_meshvar
    ):
        scan, meshvar = scan_1d_and_meshvar
        meshvar.attributes["topology_dimension"] = 2
        msg = "has topology_dimension=\"2\", but.* no 'face_node_connectivity'"
        self.check(scan, "R113", msg)

    def test_r113_mesh_topologydim0_extra_faceconn(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        # Add extra dims + a variable, to mimic face-node connectivity.
        # (which is not valid in a 0-d mesh)
        scan.dimensions["faces"] = NcDimSummary("faces", 3)
        scan.dimensions["face_vertices"] = NcDimSummary("face_vertices", 4)
        facenodes_name = "fake_face_nodes"
        scan.variables[facenodes_name] = NcVariableSummary(
            name=facenodes_name,
            dimensions=["faces", "face_vertices"],
            shape=(3, 4),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "face_node_connectivity"},
        )
        meshvar.attributes["face_node_connectivity"] = facenodes_name
        msg = (
            'has topology_dimension="0", but the presence of.*'
            "'face_node_connectivity'.*implies it should be 2."
        )
        self.check(scan, "R113", msg)

    def test_r114_mesh_topologydim1_extra_boundsconn(
        self, scan_1d_and_meshvar
    ):
        scan, meshvar = scan_1d_and_meshvar
        # Add extra dims and a variable, to mimic a bounds connectivity.
        # (which is not valid in a 1-d mesh)
        scan.dimensions["bounds"] = NcDimSummary("bounds", 3)
        scan.dimensions["bounds_ends"] = NcDimSummary("bounds_ends", 2)
        boundnodes_name = "fake_bounds"
        scan.variables[boundnodes_name] = NcVariableSummary(
            name=boundnodes_name,
            dimensions=["bounds", "bounds_ends"],
            shape=(3, 2),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "boundary_node_connectivity"},
        )
        meshvar.attributes["boundary_node_connectivity"] = boundnodes_name
        msg = (
            "has a 'boundary_node_connectivity'.*"
            "there is no 'face_node_connectivity'"
        )
        self.check(scan, "R114", msg)

    def _check_w_r305(self, scan, errcode, msg):
        # For multiple cases which generate an annoying extra R305
        statements = [
            (errcode, msg),
            (
                "R305",
                "does not contain any element dimension of the parent mesh",
            ),
        ]
        self.check(scan, statements=statements)

    def test_r115_mesh_edgedim_unknown(self, scan_1d_and_meshvar):
        scan, meshvar = scan_1d_and_meshvar
        meshvar.attributes["edge_dimension"] = "unknown_dim"
        # Avoid checking the data-variable, which now has a dim problem.
        del scan.variables["sample_data"].attributes["mesh"]
        msg = 'edge_dimension="unknown_dim".* not a dimension'
        self._check_w_r305(scan, "R115", msg)

    def test_r117_mesh_facedim_unknown(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["face_dimension"] = "unknown_dim"
        # Avoid checking the data-variable, which now has a dim problem.
        del scan.variables["sample_data"].attributes["mesh"]
        msg = 'face_dimension="unknown_dim".* not a dimension'
        self._check_w_r305(scan, "R117", msg)

    def test_r116_mesh_missing_needed_edgedim(self, scan_2d_and_meshvar):
        # Check that, if some edge connectivities have a non-standard dim
        # order, we then require an edge-dim attribute.
        scan, meshvar = scan_2d_and_meshvar

        # First add edges and a face-edges map to the 2d mesh
        # (because there are no optional edge connectivities in a 1d mesh)
        scan.dimensions["edge_dim"] = NcDimSummary("edge_dim", 3)
        scan.dimensions["n_edge_ends"] = NcDimSummary("n_edge_ends", 2)
        edgenodes_name = "edge_nodes_var"
        edgenodes_conn = NcVariableSummary(
            name=edgenodes_name,
            dimensions=["edge_dim", "n_edge_ends"],
            shape=(3, 2),
            dtype=np.dtype(np.int64),
            attributes={"cf_role": "edge_node_connectivity"},
        )
        # Now add the (optional) edge-face connectivity.
        edgeface_name = "edge_faces_var"
        edgeface_conn = NcVariableSummary(
            name=edgeface_name,
            dimensions=["edge_dim", "num_vertices"],
            shape=(6, 4),
            dtype=np.dtype(np.int64),
            attributes={"cf_role": "edge_face_connectivity"},
        )
        scan.variables[edgenodes_name] = edgenodes_conn
        scan.variables[edgeface_name] = edgeface_conn
        meshvar.attributes["edge_node_connectivity"] = edgenodes_name
        meshvar.attributes["edge_face_connectivity"] = edgeface_name

        # Check this is still ok, with no edge-dim attribute.
        assert "edge_dimension" not in meshvar.attributes
        self.check(scan)

        # Now swap the dim-order of the 'edge_faces' variable.
        edgeface_conn.dimensions = edgeface_conn.dimensions[::-1]
        # That should trigger the error
        msg = (
            r"has no 'edge_dimension' attribute.*"
            "edge connectivities with non-standard dimension order : "
            f'"{edgeface_name}"'
        )
        self.check(scan, "R116", msg)

        # Reinstating the 'edge_dimension' attribute should make it OK again.
        meshvar.attributes["edge_dimension"] = "edge_dim"
        self.check(scan)

    def test_r118_mesh_missing_needed_facedim_2(self, scan_2d_and_meshvar):
        # Check that, if some edge connectivities have a non-standard dim
        # order, we then require an edge-dim attribute.
        scan, meshvar = scan_2d_and_meshvar

        # Add a face-face map to the 2d mesh
        faceface_name = "face_face_var"
        faceface_conn = NcVariableSummary(
            name=faceface_name,
            dimensions=["face_dim", "n_vertices"],
            shape=(6, 4),
            dtype=np.dtype(np.int64),
            attributes={"cf_role": "face_face_connectivity"},
        )
        scan.variables[faceface_name] = faceface_conn
        meshvar.attributes["face_face_connectivity"] = faceface_name

        # Remove the face-dim attribute
        del meshvar.attributes["face_dimension"]
        # Check that this it is still OK, with no checking errors
        self.check(scan)

        # Now swap the dim-order of the 'face_edges' variable.
        faceface_conn.dimensions = faceface_conn.dimensions[::-1]
        # That should trigger the error
        msg = (
            r"has no 'face_dimension' attribute.*"
            "face connectivities with non-standard dimension order : "
            f'"{faceface_name}"'
        )
        self.check(scan, "R118", msg)

        # Reinstating the 'face_dimension' attribute should make it OK again.
        meshvar.attributes["face_dimension"] = "face_dim"
        self.check(scan)

    def test_r119_mesh_faceface_no_faces(self, scan_1d_and_meshvar):
        scan, meshvar = scan_1d_and_meshvar
        # Add extra dims + a variable to mimic a 'face-face' connectivity.
        # (which is not valid in a 1-d mesh)
        scan.dimensions["faces"] = NcDimSummary("faces", 3)
        scan.dimensions["face_N_faces"] = NcDimSummary("face_N_faces", 4)
        faceface_name = "fake_face_faces"
        scan.variables[faceface_name] = NcVariableSummary(
            name=faceface_name,
            dimensions=["faces", "face_N_faces"],
            shape=(3, 4),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "face_face_connectivity"},
        )
        meshvar.attributes["face_face_connectivity"] = faceface_name
        msg = (
            "has a 'face_face_connectivity'.*"
            "there is no 'face_node_connectivity'"
        )
        self._check_w_r305(scan, "R119", msg)

    def test_r120_mesh_faceedge_no_faces(self, scan_1d_and_meshvar):
        scan, meshvar = scan_1d_and_meshvar
        # Add extra dims + a variable, to mimic face-edge connectivity.
        # (which is not valid for a 1-d mesh)
        scan.dimensions["faces"] = NcDimSummary("faces", 5)
        scan.dimensions["face_N_edges"] = NcDimSummary("face_N_edges", 3)
        faceedges_name = "fake_face_edges"
        scan.variables[faceedges_name] = NcVariableSummary(
            name=faceedges_name,
            dimensions=["faces", "face_N_edges"],
            shape=(5, 3),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "face_edge_connectivity"},
        )
        meshvar.attributes["face_edge_connectivity"] = faceedges_name
        msg = (
            "has a 'face_edge_connectivity'.*"
            "there is no 'face_node_connectivity'"
        )
        self._check_w_r305(scan, "R120", msg)

    def test_r120_mesh_faceedge_no_edges(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar = scan.variables["topology"]
        # Add extra dims + a variable, to mimic face-edge connectivity.
        # (which is not valid, as the mesh has no edges)
        scan.dimensions["face_N_edges"] = NcDimSummary("face_N_edges", 3)
        faceedges_name = "fake_face_edges"
        scan.variables[faceedges_name] = NcVariableSummary(
            name=faceedges_name,
            dimensions=["faces", "face_N_edges"],
            shape=(6, 3),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "face_edge_connectivity"},
        )
        meshvar.attributes["face_edge_connectivity"] = faceedges_name
        msg = (
            "has a 'face_edge_connectivity'.*"
            "there is no 'edge_node_connectivity'"
        )
        self._check_w_r305(scan, "R120", msg)

    def test_r121_mesh_edgeface_no_faces(self, scan_1d_and_meshvar):
        scan, meshvar = scan_1d_and_meshvar
        # Add extra dims + a variable, to mimic edge-face connectivity.
        # (which is not valid for a 1-d mesh)
        scan.dimensions["edges"] = NcDimSummary("edges", 5)
        scan.dimensions["edge_N_faces"] = NcDimSummary("edge_N_faces", 3)
        edgefaces_name = "fake_edge_faces"
        scan.variables[edgefaces_name] = NcVariableSummary(
            name=edgefaces_name,
            dimensions=["edges", "edge_N_faces"],
            shape=(5, 3),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "edge_face_connectivity"},
        )
        meshvar.attributes["edge_face_connectivity"] = edgefaces_name
        msg = (
            "has a 'edge_face_connectivity'.*"
            "there is no 'face_node_connectivity'"
        )
        self._check_w_r305(scan, "R121", msg)

    def test_r121_mesh_edgeface_no_edges(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        # Add extra dims + a variable, to mimic edge-face connectivity.
        # (which is not valid as the mesh has no edges)
        scan.dimensions["edges"] = NcDimSummary("edges", 5)
        scan.dimensions["edge_N_faces"] = NcDimSummary("edge_N_faces", 3)
        edgefaces_name = "fake_edge_faces"
        scan.variables[edgefaces_name] = NcVariableSummary(
            name=edgefaces_name,
            dimensions=["edges", "edge_N_faces"],
            shape=(5, 3),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "edge_face_connectivity"},
        )
        meshvar.attributes["edge_face_connectivity"] = edgefaces_name
        msg = (
            "has a 'edge_face_connectivity'.*"
            "there is no 'edge_node_connectivity'"
        )
        self._check_w_r305(scan, "R121", msg)

    def test_a101_mesh_variable_dimensions(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        meshvar.dimensions = ["num_node"]
        meshvar.shape = (8,)  # to make consistent, but probably unnecessary?
        self.check(scan, "A101", 'Mesh variable "topology" has dimensions')

    def test_a102_mesh_variable_stdname(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        meshvar.attributes["standard_name"] = "air_temperature"
        self.check(
            scan,
            "A102",
            "Mesh variable \"topology\" has a 'standard_name' attribute",
        )

    def test_a103_mesh_variable_units(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        meshvar.attributes["units"] = "K"
        self.check(
            scan,
            "A103",
            "Mesh variable \"topology\" has a 'units' attribute",
        )

    # Note: A104 is tested in TestChecker_Dataset

    def test_a105_mesh_invalid_boundarydim(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["boundary_dimension"] = "odd_name"
        msg = (
            "has an attribute 'boundary_dimension', which is "
            "not a valid UGRID term"
        )
        self.check(scan, "A106", msg)

    def test_a105_mesh_invalid_nodedim(self, scan_2d_and_meshvar):
        scan, meshvar = scan_2d_and_meshvar
        meshvar.attributes["node_dimension"] = "odd_name"
        msg = (
            "has an attribute 'node_dimension', which is "
            "not a valid UGRID term"
        )
        self.check(scan, "A106", msg)

    def test_a106_mesh_unwanted_edgedim(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        meshvar.attributes["edge_dimension"] = "odd_name"
        msg = (
            "has an attribute 'edge_dimension', which is not valid.*"
            r"no 'edge_node_connectivity'\."
        )
        self.check(scan, "R123", msg)

    def test_a106_mesh_unwanted_facedim(self, scan_0d_and_meshvar):
        scan, meshvar = scan_0d_and_meshvar
        meshvar.attributes["face_dimension"] = "odd_name"
        msg = (
            "has an attribute 'face_dimension', which is not valid.*"
            r"no 'face_node_connectivity'\."
        )
        self.check(scan, "R122", msg)


class TestChecker_DataVariables(DatasetChecker):
    @fixture
    def scan_2d_and_datavar(self, scan_2d_mesh):
        return scan_2d_mesh, scan_2d_mesh.variables["sample_data"]

    def test_r501_datavar_mesh_with_lis(self, scan_2d_mesh):
        # Add an lis and convert the original var to it.
        self._convert_to_lis(scan_2d_mesh)
        # The data-var was modified to reference the lis + its dim.
        # Reinstate the original mesh-basis, but leave the lis reference in
        # place to trigger the intended error.
        data_var = scan_2d_mesh.variables["sample_data"]
        assert str(data_var.attributes["location_index_set"]) == "lis"
        data_var.attributes["mesh"] = "topology"
        data_var.attributes["location"] = "face"
        assert data_var.dimensions == ("lis_faces",)
        data_var.dimensions = ("face_dim",)
        msg = (
            "\"sample_data\" has a 'location_index_set' attribute.*"
            r"invalid since it is based on a 'mesh' attribute\."
        )
        self.check(scan_2d_mesh, "R501", msg)

    def test_r502_datavar_empty_mesh(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = ""
        msg = r'mesh="", which is not a single variable name\.'
        self.check(scan, "R502", msg)

    def test_r502_datavar_invalid_mesh_name(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = "/bad/"
        msg = r'mesh="/bad/", which is not a valid variable name\.'
        self.check(scan, "R502", msg)

    def test_r502_datavar_bad_mesh_dtype(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = 2.0
        msg = r'mesh="2.0", which is not a string value\.'
        self.check(scan, "R502", msg)

    def test_r502_datavar_multi_mesh_name(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = "this that other"
        msg = r'mesh="this that other", which is not a single variable name\.'
        self.check(scan, "R502", msg)

    def test_r502_datavar_missing_meshvar(self, scan_2d_and_datavar):
        scan, datavar = scan_2d_and_datavar
        datavar.attributes["mesh"] = "absent"
        msg = r'mesh="absent", which is not a variable in the dataset\.'
        self.check(scan, "R502", msg)

    def test_r503_datavar_mesh_no_location(self, scan_0d_mesh):
        meshdata_var = scan_0d_mesh.variables["sample_data"]
        del meshdata_var.attributes["location"]
        msg = r"\"sample_data\" has no 'location' attribute\."
        self.check(scan_0d_mesh, "R503", msg)

    def test_r504_datavar_mesh_invalid_location(self, scan_0d_mesh):
        meshdata_var = scan_0d_mesh.variables["sample_data"]
        meshdata_var.attributes["location"] = "other"
        msg = (
            '"sample_data" has location="other", '
            r'which is not one of "face", "edge" or "node"\.'
        )
        self.check(scan_0d_mesh, "R504", msg)

    def test_r505_datavar_mesh_nonexistent_location(self, scan_0d_mesh):
        meshdata_var = scan_0d_mesh.variables["sample_data"]
        meshdata_var.attributes["location"] = "edge"
        msg = (
            '"sample_data" has location="edge", which is a location that '
            r'does not exist in the parent mesh, "topology"\.'
        )
        self.check(scan_0d_mesh, "R505", msg)

    @fixture
    def scan_0d_with_lis_datavar(self, scan_0d_mesh):
        # Produce a modified version of the 0d mesh that "interposes" a
        # location-index-set between the "sample_data" var and the mesh.
        self._convert_to_lis(scan_0d_mesh)
        return scan_0d_mesh, scan_0d_mesh.variables["sample_data"]

    def test_r506_datavar_lis_with_mesh(self, scan_0d_with_lis_datavar):
        # An lis datavar should not have a 'mesh' attribute.
        scan, data_var = scan_0d_with_lis_datavar
        data_var.attributes["mesh"] = "topology"
        msg = (
            "\"sample_data\" has a 'mesh' attribute.*"
            "invalid since it is based on a 'location_index_set' attribute."
        )
        self.check(scan, "R506", msg)

    def test_r507_datavar_lis_with_location(self, scan_0d_with_lis_datavar):
        # An lis datavar should not have a 'location' attribute.
        scan, data_var = scan_0d_with_lis_datavar
        data_var.attributes["location"] = "node"
        msg = (
            "\"sample_data\" has a 'location' attribute.*"
            "invalid since it is based on a 'location_index_set' attribute."
        )
        self.check(scan, "R507", msg)

    # NOTE: only do one of these, as the valid-reference checking mechanism is
    # tested elsewhere : TestChecker_Coords.test_xxx_coord_bounds_yyy
    def test_r508_datavar_lis_invalid(self, scan_0d_with_lis_datavar):
        # The lis attribute should be a valid variable reference.
        scan, data_var = scan_0d_with_lis_datavar
        data_var.attributes["location_index_set"] = "/bad/"
        msg = (
            r'location_index_set="/bad/", '
            r"which is not a valid variable name\."
        )
        self.check(scan, "R508", msg)

    def test_r509_datavar_lis_no_meshdims(self, scan_0d_with_lis_datavar):
        scan, data_var = scan_0d_with_lis_datavar
        assert data_var.dimensions == ("lis_nodes",)
        data_var.dimensions = ("num_ends",)
        msg = (
            r"dimensions \('num_ends',\), of which "
            r"0 are mesh dimensions, instead of 1\."
        )
        self.check(scan, "R509", msg)

    def test_r509_datavar_lis_multi_meshdims(self, scan_0d_with_lis_datavar):
        scan, data_var = scan_0d_with_lis_datavar
        assert data_var.dimensions == ("lis_nodes",)
        data_var.dimensions = ("lis_nodes", "lis_nodes")
        msg = (
            r"dimensions \('lis_nodes', 'lis_nodes'\), of which "
            r"2 are mesh dimensions, instead of 1\."
        )
        self.check(scan, "R509", msg)

    def test_r510_datavar_mesh_wrongdim(self, scan_2d_and_datavar):
        scan, data_var = scan_2d_and_datavar
        # Add edges so we have 2 different element dimensions to work with
        self._add_edges(scan)
        # Set datavar element dimension different to the parent mesh location
        assert data_var.dimensions == ("face_dim",)
        data_var.dimensions = ("edges",)
        msg = (
            'element dimension "edges", which does not match the '
            'face dimension of the "topology" mesh, '
            r'which is "face_dim"\.'
        )
        self.check(scan, "R510", msg)

    def test_r510_datavar_lis_wrongdim(self, scan_2d_and_datavar):
        scan, data_var = scan_2d_and_datavar
        self._convert_to_lis(scan)
        # Set datavar element dimension different to the parent mesh location
        assert data_var.dimensions == ("lis_faces",)
        data_var.dimensions = ("num_node",)
        msg = (
            'element dimension "num_node", which does not match '
            'the face dimension of the "lis" location_index_set, '
            r'which is "lis_faces"\.'
        )
        self.check(scan, "R510", msg)


class TestChecker_Coords(DatasetChecker):
    @fixture
    def scan_2d_and_coordvar(self, scan_2d_mesh):
        return scan_2d_mesh, scan_2d_mesh.variables["node_lon"]

    # Test mesh-coordinate checking.
    def test_r201_coord_multidim(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        coord.dimensions = coord.dimensions + ("num_vertices",)
        msg = '"node_lon".*one dimension.*has 2 dimensions'
        self.check(scan, "R201", msg)

    def test_r202_coord_wrongdim(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        coord.dimensions = ("face_dim",)
        msg = (
            'dimension "face_dim", but the parent mesh '
            'node dimension is "num_node".'
        )
        self.check(scan, "R202", msg)

    def test_r203_coord_bounds_multiple_names(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Add an invalid bounds attribute to the node_lon coord.
        coord.attributes["bounds"] = "a b"
        msg = (
            '"node_lon" within topology:node_coordinates has bounds="a b", '
            "which is not a single variable"
        )
        self.check(scan, "R203", msg)

    def test_r203_coord_bounds_bad_attrdtype(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Add an invalid bounds attribute to the node_lon coord.
        coord.attributes["bounds"] = 2.0
        msg = r'has bounds="2.0", which is not a string value\.'
        self.check(scan, "R203", msg)

    def test_r203_coord_bounds_bad_name(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Add an invalid bounds attribute to the node_lon coord.
        coord.attributes["bounds"] = "/bad/"
        msg = (
            r'"node_lon" within topology:node_coordinates has bounds="/bad/", '
            r"which is not a valid variable name\."
        )
        self.check(scan, "R203", msg)

    def test_r203_coord_bounds_missing_var(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Add an invalid bounds attribute to the node_lon coord.
        coord.attributes["bounds"] = "other_var"
        msg = (
            '"node_lon" within topology:node_coordinates has '
            'bounds="other_var", which is not a variable in the dataset'
        )
        self.check(scan, "R203", msg)

    def test_r203_coord_bounds_missing_element_dim(self, scan_2d_mesh):
        scan = scan_2d_mesh
        coord = scan.variables["longitude"]
        # Add face-lon bounds, with inappropriate dims.
        # N.B. match the expected dim *lengths*, to avoid additional errors
        scan.dimensions["extra1"] = NcDimSummary("extra1", 6)
        scan.dimensions["extra2"] = NcDimSummary("extra2", 4)
        facelon_bds_name = "node_lons_bounds"
        scan.variables[facelon_bds_name] = NcVariableSummary(
            name=facelon_bds_name,
            dimensions=("extra1", "extra2"),
            shape=(6, 4),
            dtype=coord.dtype,
        )
        coord.attributes["bounds"] = facelon_bds_name
        msg = (
            r"dimensions \('extra1', 'extra2'\).*"
            'does not include the parent variable dimension, "face_dim".'
        )
        self.check(scan, "R203", msg)

    def test_r203_coord_bounds_bad_ndims(self, scan_2d_mesh):
        coord = scan_2d_mesh.variables["longitude"]
        # Add node-lon bounds, with wrong n-dims (but does include the parent).
        facelons_bds_name = "face_lons_bounds"
        scan_2d_mesh.variables[facelons_bds_name] = NcVariableSummary(
            name=facelons_bds_name,
            dimensions=("face_dim",),
            shape=(6,),
            dtype=coord.dtype,
        )
        coord.attributes["bounds"] = facelons_bds_name
        msg = r"dimensions \('face_dim',\).* should be 2, instead of 1\."
        self.check(
            scan_2d_mesh,
            statements=[
                ("R203", msg),
                # N.B. can't help getting this one too
                ("A205", "does not match the shape"),
            ],
        )

    def test_r203_coord_bounds_stdname_clash(self, scan_2d_mesh):
        coord = scan_2d_mesh.variables["longitude"]
        # Add node-lon bounds, with wrong n-dims (but does include the parent).
        facelons_bds_name = "face_lons_bounds"
        scan_2d_mesh.variables[facelons_bds_name] = NcVariableSummary(
            name=facelons_bds_name,
            dimensions=("face_dim", "num_vertices"),
            shape=(6, 4),
            dtype=coord.dtype,
            attributes={"standard_name": "junk"},
        )
        coord.attributes["bounds"] = facelons_bds_name
        msg = (
            'standard_name="junk", which does not match the parent '
            r'\'standard_name\' of "longitude"\.'
        )
        self.check(scan_2d_mesh, "R203", msg)

    def test_r203_coord_bounds_stdname_noparentstdname(self, scan_2d_mesh):
        coord = scan_2d_mesh.variables["longitude"]
        # Add node-lon bounds, with wrong n-dims (but does include the parent).
        facelons_bds_name = "face_lons_bounds"
        scan_2d_mesh.variables[facelons_bds_name] = NcVariableSummary(
            name=facelons_bds_name,
            dimensions=("face_dim", "num_vertices"),
            shape=(6, 4),
            dtype=coord.dtype,
            attributes={"standard_name": coord.attributes["standard_name"]},
        )
        coord.attributes["bounds"] = facelons_bds_name
        # Fix so the bounds has a standard-name, but the parent does not
        del coord.attributes["standard_name"]
        msg = (
            'standard_name="longitude", which does not match the parent '
            r'\'standard_name\' of "<none>"\.'
        )
        # N.B. this causes an additional A203 statement, which can't be avoided
        msg2 = (
            'coordinate variable "longitude".*'
            r"has no \'standard_name\' attribute\."
        )
        self.check(scan_2d_mesh, statements=[("R203", msg), ("A203", msg2)])

    def test_r203_coord_bounds_units_clash(self, scan_2d_mesh):
        coord = scan_2d_mesh.variables["longitude"]
        # Add node-lon bounds, with wrong n-dims (but does include the parent).
        facelons_bds_name = "face_lons_bounds"
        scan_2d_mesh.variables[facelons_bds_name] = NcVariableSummary(
            name=facelons_bds_name,
            dimensions=("face_dim", "num_vertices"),
            shape=(6, 4),
            dtype=coord.dtype,
            attributes={"units": "junk"},
        )
        coord.attributes["bounds"] = facelons_bds_name
        msg = (
            'units="junk", which does not match the parent '
            r'\'units\' of "degrees_east"\.'
        )
        self.check(scan_2d_mesh, "R203", msg)

    def test_r203_coord_bounds_units_noparentunits(self, scan_2d_mesh):
        coord = scan_2d_mesh.variables["longitude"]
        # Add node-lon bounds, with wrong n-dims (but does include the parent).
        facelons_bds_name = "face_lons_bounds"
        scan_2d_mesh.variables[facelons_bds_name] = NcVariableSummary(
            name=facelons_bds_name,
            dimensions=("face_dim", "num_vertices"),
            shape=(6, 4),
            dtype=coord.dtype,
            attributes={"units": "degrees_east"},
        )
        coord.attributes["bounds"] = facelons_bds_name
        # Fix so the bounds has a standard-name, but the parent does not
        del coord.attributes["units"]
        msg = (
            'units="degrees_east", which does not match the parent '
            r'\'units\' of "<none>"\.'
        )
        # N.B. this causes an additional A204 statement, which can't be avoided
        msg2 = r'coordinate.* "longitude".* has no \'units\' attribute\.'
        self.check(scan_2d_mesh, statements=[("R203", msg), ("A204", msg2)])

    #
    # Advisory warnings
    #

    def test_a201_coord_multiple_meshes(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Copy 'topology' mesh and all its components to make 'topology_2'
        mesh2 = next_mesh(scan, "topology")
        # Reference the old 'node_lon' variable in the new 'topology_2' mesh
        assert mesh2.attributes["node_coordinates"] == "node_lat_2 node_lon_2"
        mesh2.attributes["node_coordinates"] = "node_lat_2 node_lon"
        msg = (
            '"node_lon" is referenced by multiple mesh variables : '
            "topology:node_coordinates, topology_2:node_coordinates."
        )
        # Note: we also get a '2 meshes using 1 dim' error
        # - this can't be avoided.
        msg2 = (
            'has dimension "num_node", but the parent mesh node dimension '
            'is "num_node_2"'
        )
        self.check(scan, statements=[("A201", msg), ("R202", msg2)])

    def test_a202_coord_not_floating_point(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        coord.dtype = np.dtype(np.int32)
        msg = (
            'variable "node_lon" within topology:node_coordinates has '
            r'type "int32", which is not a floating-point type\.'
        )
        self.check(scan, "A202", msg)

    def test_a203_coord_no_stdname(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        del coord.attributes["standard_name"]
        msg = r"has no 'standard_name' attribute\."
        self.check(scan, "A203", msg)

    def test_a204_coord_no_units(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        del coord.attributes["units"]
        msg = r"has no 'units' attribute\."
        self.check(scan, "A204", msg)

    def test_a205_data_bounds_conns_mismatch(self, scan_mini_w_data):
        scan = scan_mini_w_data
        # Add a face-coordinate, and a face-coordinate bounds variable.
        scan.variables["face_x"] = NcVariableSummary(
            name="face_x",
            dimensions=("faces",),
            shape=(2,),
            dtype=np.dtype(float),
            data=FakeVariableDataWrapper(None, np.arange(2.0)),
            attributes={
                "standard_name": "longitude",
                "units": "degrees_east",
                "bounds": "face_x_bds",
            },
        )
        scan.variables["topology"].attributes["face_coordinates"] = "face_x"
        scan.dimensions["face_bnds"] = NcDimSummary("face_bnds", 4)
        bounds_points = [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 99.0]]
        bounds_points = np.ma.masked_greater(bounds_points, 10)
        bdsvar = NcVariableSummary(
            name="face_x_bds",
            dimensions=("faces", "face_bds"),
            shape=(2, 4),  # N.B. must match the face_nodes connectivity shape
            dtype=np.dtype(float),
            data=FakeVariableDataWrapper(None, bounds_points),
        )
        scan.variables["face_x_bds"] = bdsvar

        # This should work OK
        self.check_withdata(scan)

        # Disturb the masked point : it should not matter.
        bdsvar.data.array[1, -1] = 999.9
        self.check_withdata(scan)

        # Disturb a valid point : this should now fail.
        bdsvar.data.array[1, 0] = 999.9
        msg = (
            'has bounds="face_x_bds", which does not match the expected values '
            'calculated from the "node_x" node coordinate and the face '
            'connectivity, "face_nodes".'
        )
        self.check_withdata(scan, "A205", msg)

    def test_a206_nodes_with_bounds(self, scan_2d_and_coordvar):
        scan, coord = scan_2d_and_coordvar
        # Add a bounds var to the node coordinates.
        scan.variables["nodelon_bounds"] = NcVariableSummary(
            name="node_lon_bounds",
            dimensions=("num_node", "num_vertices"),
            shape=(8, 4),  # N.B. must match
            dtype=np.dtype(float),
        )
        coord.attributes["bounds"] = "nodelon_bounds"
        msg = (
            "'bounds' attribute, which is not valid for a "
            "coordinate of location 'node'."
        )
        self.check(scan, "A206", msg)


class TestChecker_Connectivities(DatasetChecker):
    @fixture
    def scan_2d_and_connvar(self, scan_2d_mesh):
        return scan_2d_mesh, scan_2d_mesh.variables["face_nodes"]

    def test_r301_conn_missing_cfrole(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        del conn.attributes["cf_role"]
        msg = "\"face_nodes\".* has no 'cf_role' attribute."
        self.check(scan, "R301", msg)

    def test_r302_conn_bad_cfrole(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        # Set to a valid CF discrete-sampling term, to avoid a warning about
        # valid CF values (a possible future check)
        conn.attributes["cf_role"] = "profile_id"
        msg = (
            '"face_nodes".* has cf_role="profile_id".*'
            "not a valid UGRID connectivity"
        )
        self.check(scan, "R302", msg)

    def test_r303_conn_cfrole_differentfromparent(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.attributes["cf_role"] = "face_face_connectivity"
        msg = (
            'cf_role="face_face_connectivity".*'
            "different from its role in the parent.*"
            '"face_node_connectivity"'
        )
        self.check(scan, "R303", msg)

    def test_r304_conn_bad_ndims(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.dimensions = conn.dimensions + ("num_vertices",)
        msg = (
            r"dimensions \('face_dim', 'num_vertices', 'num_vertices'\), "
            "of which there are 3, instead of 2"
        )
        self.check(scan, "R304", msg)

    def test_r305_conn_no_parent_dim(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.dimensions = ("num_vertices", "num_vertices")
        msg = (
            r"dimensions \('num_vertices', 'num_vertices'\).*"
            "does not contain any element dimension of the parent mesh"
        )
        self.check(scan, "R305", msg)

    def test_r306_conn_no_nonparent_dim(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.dimensions = ("face_dim", "face_dim")
        msg = (
            r"dimensions \('face_dim', 'face_dim'\).*"
            "does not contain any dimension which is "
            "not an element dimension of the parent mesh"
        )
        self.check(scan, "R306", msg)

    def test_r307_conn_wrong_element_dim(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.dimensions = ("num_node", "num_vertices")
        msg = (
            "does not include the expected face dimension "
            r'of the parent mesh, "face_dim"\.'
        )
        self.check(scan, "R307", msg)

    def test_r308_conn_bad_num_edge_ends(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        self._add_edges(scan)
        # Adjust the length of the 'n_edge_ends' dim : this is the error.
        scan.dimensions["n_edge_ends"] = NcDimSummary("n_edge_ends", 7)
        msg = (
            '"edge_nodes".* contains the non-mesh dimension "n_edge_ends", '
            r"but this has length 7 instead of 2\."
        )
        self.check(scan, "R308", msg)

    def test_r309_conn_bad_startindex_value(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.attributes["start_index"] = np.array(3, dtype=conn.dtype)
        msg = (
            '"face_nodes".* has start_index="3", '
            r"which is not either 0 or 1\."
        )
        self.check(scan, "R309", msg)

    def test_r310_required_indices_missing(self, scan_mini_w_data):
        scan = scan_mini_w_data
        # Add some edge data for this check, as it doesn't apply to face-nodes.
        self._add_edges(scan, n_edges=2)
        edge_conn = scan.variables["edge_nodes"]
        # Replace the var data with a 'wrapped' form, to set edge-node indices
        edge_conn.data = FakeVariableDataWrapper(
            edge_conn.data, [[0, 1], [2, 3]]
        )
        # test with data checks : should be OK at this point
        self.check_withdata(scan)

        # Now add a missing point in edge-nodes, which should raise an error
        # N.B. *don't* use a _FillValue, since that is also forbidden
        data = edge_conn.data.array
        data = np.ma.masked_array(data)
        data[-1, -1] = np.ma.masked
        edge_conn.data.array = data
        msg = (
            "contains missing indices, which is not permitted for a "
            'connectivity of type "edge_node_connectivity".'
        )
        self.check_withdata(
            scan,
            statements=[
                (
                    "R310",
                    msg,
                ),
                # Can't easily avoid getting this one also
                ("A305", "no '_FillValue'"),
            ],
        )

    def test_r311_subtriangle_faces(self, scan_mini_w_data):
        scan = scan_mini_w_data
        face_conn = scan.variables["face_nodes"]
        # Wrap the data array to give it different values
        modified_array = face_conn.data.fetch_array()
        modified_array[1, 0] = np.ma.masked
        face_conn.data = FakeVariableDataWrapper(
            face_conn.data, modified_array
        )
        self.check_withdata(scan, "R311", "faces with less than 3 vertices")

    def test_r311_nonstandard_dimorder(self, scan_mini_w_data):
        # Check that the R311 test still works with nonstandard dim order
        scan = scan_mini_w_data
        face_conn = scan.variables["face_nodes"]

        # Transpose the face-nodes connectivity
        array = face_conn.data.fetch_array()
        face_conn.data = FakeVariableDataWrapper(
            face_conn.data, array.transpose()
        )  # install a modified array
        face_conn.dimensions = face_conn.dimensions[::-1]  # fix var dims
        face_conn.shape = face_conn.shape[::-1]  # fix var shape
        # also needs a 'face_dimension' attribute to avoid misinterpretation
        scan.variables["topology"].attributes["face_dimension"] = "faces"

        # That should check out OK
        self.check_withdata(scan)

        # Reduce the quad face to 2 points
        assert face_conn.shape == (4, 2)
        assert np.all(face_conn.data.fetch_array()[:, 0] == [1, 2, 3, 4])
        face_conn.data.array[:2, 0] = np.ma.masked
        # This should now raise an error
        self.check_withdata(scan, "R311", "faces with less than 3 vertices")

    #
    # Advisory checks
    #

    def test_a301_conn_multiple_meshes(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar

        # Copy 'topology' mesh and all its components to make 'topology_2'
        mesh2 = next_mesh(scan, "topology")
        # Reference the old 'face_nodes' variable in the new 'topology_2' mesh
        assert mesh2.attributes["face_node_connectivity"] == "face_nodes_2"
        mesh2.attributes["face_node_connectivity"] = "face_nodes"
        # Also delete the 'orphan', to avoid further errors
        del scan.variables["face_nodes_2"]
        msg = (
            '"face_nodes" is referenced by multiple mesh variables : '
            "topology:face_node_connectivity, "
            r"topology_2:face_node_connectivity\."
        )
        # Note: we also get a 'doesn't match parent dim' error
        # - this can't easily be avoided.
        msg2 = "does not contain any element dimension of the parent mesh"
        self.check(scan, statements=[("A301", msg), ("R305", msg2)])

    def test_a301_conn_nomesh(self, scan_2d_and_connvar):
        # Test the 'orphan connectivity' detection.
        scan, conn = scan_2d_and_connvar
        # Add edges to the scan.
        self._add_edges(scan)
        # But 'orphan' the edges connectivity by removing it from the mesh.
        meshvar = scan.variables["topology"]
        del meshvar.attributes["edge_node_connectivity"]
        self.check(scan, "A301", '"edge_nodes" has no parent mesh.')

    def test_a302_conn_nonintegraltype(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.dtype = np.dtype(np.float32)
        # N.B. also adjust dtype of start_index, to avoid an A303
        conn.attributes["start_index"] = np.array(0.0, dtype=conn.dtype)
        msg = (
            '"face_nodes".* has type "float32", '
            r"which is not an integer type\."
        )
        self.check(scan, "A302", msg)

    def test_a303_conn_bad_startindex_type(self, scan_2d_and_connvar):
        scan, conn = scan_2d_and_connvar
        conn.attributes["start_index"] = np.array(1.0)
        msg = (
            "a 'start_index' of type \"float64\".* different from the "
            'variable type, "int32"'
        )
        self.check(scan, "A303", msg)

    def test_a304_conn_unexpected_fillvalue(self, scan_2d_and_connvar):
        # Some connectivities should never have a fill-value.
        # Previously applied to *any* "xxx_node_connectivity", which is wrong:
        # should be edge-/boundary-node, but *not* face-node
        scan, conn = scan_2d_and_connvar
        self._add_edges(scan)
        conn.attributes["_FillValue"] = np.array(-1, dtype=conn.dtype)
        # This one is OK because it is needed for a 'flexible' 2d mesh ...
        self.check(scan)

        # ... But an 'edge_node_connectivity' should *not* have a fill-value.
        conn2 = scan.variables["edge_nodes"]
        conn2.attributes["_FillValue"] = np.array(-1, dtype=conn.dtype)
        # This will raise an A304
        msg = (
            "\"edge_nodes\".* has a '_FillValue' attribute, which "
            "should not be present on a "
            r'"edge_node_connectivity" connectivity\.'
        )
        self.check(scan, "A304", msg)

    def test_a305_missing_without_fillvalue(self, scan_mini_w_data):
        scan = scan_mini_w_data
        facenodes_var = scan.variables["face_nodes"]
        del facenodes_var.attributes["_FillValue"]
        msg = "contains missing indices, but has no '_FillValue' attribute"
        self.check_withdata(scan, "A305", msg)

    def _add_edges_and_faceedges(self, scan):
        self._add_edges(scan)
        faceedge_varname = "face_edges"
        scan.variables[faceedge_varname] = NcVariableSummary(
            name=faceedge_varname,
            dimensions=["face_dim", "n_vertices"],
            shape=(6, 4),
            dtype=np.dtype(np.int32),
            attributes={"cf_role": "face_edge_connectivity"},
        )
        meshvar = scan.variables["topology"]
        meshvar.attributes["face_edge_connectivity"] = faceedge_varname

    def test_a306_conn_fillvalue_baddtype(self, scan_2d_mesh):
        # Add edges and a face-edge connectivity, because face_nodes can't have
        # missing indices.
        self._add_edges_and_faceedges(scan_2d_mesh)
        conn = scan_2d_mesh.variables["face_edges"]
        conn.attributes["_FillValue"] = np.array(-1, dtype=np.int16)
        msg = (
            '"face_edges".* has a \'_FillValue\' of type "int16", '
            r'which is different from the variable type, "int32"\.'
        )
        self.check(scan_2d_mesh, "A306", msg)

    def test_a307_conn_fillvalue_badvalue(self, scan_2d_mesh):
        # Add edges and a face-edge connectivity, because face_nodes can't have
        # missing indices.
        self._add_edges_and_faceedges(scan_2d_mesh)
        conn = scan_2d_mesh.variables["face_edges"]
        conn.attributes["_FillValue"] = np.array(999, dtype=conn.dtype)
        msg = r'"face_edges".* has _FillValue="999", which is not negative\.'
        self.check(scan_2d_mesh, "A307", msg)

    def test_a308_bad_min_index(self, scan_mini_w_data):
        # Check for when an index is less than start-index
        scan = scan_mini_w_data
        facenodes_var = scan.variables["face_nodes"]
        assert facenodes_var.attributes["start_index"] == 1
        facenodes = facenodes_var.data.fetch_array()
        assert np.all(facenodes[1] == [4, 3, 5, -1])
        facenodes[1, 0] = 0
        facenodes_var.data = FakeVariableDataWrapper(
            facenodes_var.data, facenodes
        )
        msg = (
            "minimum index value of 0, "
            "which is less than its start-index of 1"
        )
        self.check_withdata(scan, "A308", msg)

    def test_a308_bad_min_index_nostartindex(self, scan_mini_w_data):
        # As previous, but using a DEFAULT start-index
        scan = scan_mini_w_data

        # remove start-index and reduce all indices to 0-based.
        facenodes_var = scan.variables["face_nodes"]
        del facenodes_var.attributes["start_index"]
        facenodes = facenodes_var.data.fetch_array()
        facenodes.data[:] = facenodes.data - 1  # NB mask remains the same
        facenodes_var.data = FakeVariableDataWrapper(
            facenodes_var.data, facenodes
        )

        # As it stands, that should be OK
        self.check_withdata(scan)

        # Now install a bad minimum index
        facenodes[1, 0] = -5
        msg = (
            "minimum index value of -5, "
            "which is less than its start-index of 0"
        )
        self.check_withdata(scan, "A308", msg)

    def test_a308_bad_max_index(self, scan_mini_w_data):
        # Check for when an index is outside the dimension range
        scan = scan_mini_w_data
        facenodes_var = scan.variables["face_nodes"]
        assert facenodes_var.attributes["start_index"] == 1
        facenodes = facenodes_var.data.fetch_array()
        assert np.all(facenodes[1] == [4, 3, 5, -1])
        facenodes[1, 0] = 6
        facenodes_var.data = FakeVariableDataWrapper(
            facenodes_var.data, facenodes
        )
        msg = (
            "contains a maximum index value of 6, "
            "which is outside the range of the "
            'relevant "nodes" dimension, 1..5.'
        )
        self.check_withdata(scan, "A308", msg)

    def test_a308_bad_max_index_nostartindex(self, scan_mini_w_data):
        # As previous, but using a DEFAULT start-index
        scan = scan_mini_w_data

        # remove start-index and reduce all indices to 0-based.
        facenodes_var = scan.variables["face_nodes"]
        del facenodes_var.attributes["start_index"]
        facenodes = facenodes_var.data.fetch_array()
        facenodes.data[:] = facenodes.data - 1  # NB mask remains the same
        facenodes_var.data = FakeVariableDataWrapper(
            facenodes_var.data, facenodes
        )

        # this should all be OK
        self.check_withdata(scan)

        # add an invalid maximum index value
        facenodes_var.data.array[1, 0] = 5

        # should now fail.
        msg = (
            "contains a maximum index value of 5, "
            "which is outside the range of the "
            'relevant "nodes" dimension, 0..4.'
        )
        self.check_withdata(scan, "A308", msg)


class TestChecker_LocationIndexSets(DatasetChecker):
    @fixture
    def scan_0d_and_lis(self, scan_0d_mesh):
        self._convert_to_lis(scan_0d_mesh)
        return scan_0d_mesh, scan_0d_mesh.variables["lis"]

    def test_basic_scan_ok(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        self.check(scan)

    def test_r401_conn_missing_cfrole(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        del lis_var.attributes["cf_role"]
        self.check(scan, "R401", r'"lis" has no \'cf_role\' attribute\.')

    def test_r401_conn_bad_cfrole(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        # Set a bad cf_role.  Use a valid CF one, to avoid any further error.
        lis_var.attributes["cf_role"] = "profile_id"
        msg = (
            '"lis" has cf_role="profile_id", '
            r'instead of "location_index_set"\.'
        )
        self.check(scan, "R401", msg)

    def test_r402_conn_missing_mesh(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        del lis_var.attributes["mesh"]
        self.check(scan, "R402", r'"lis" has no \'mesh\' attribute\.')

    def test_r402_conn_bad_mesh(self, scan_0d_and_lis):
        # Just test *one* of the reasons handled by 'var_ref_problem'
        scan, lis_var = scan_0d_and_lis
        lis_var.attributes["mesh"] = "not_included"
        msg = (
            '"lis" has mesh="not_included", which '
            r"is not a variable in the dataset\."
        )
        self.check(scan, "R402", msg)

    def test_r403_conn_missing_location(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        del lis_var.attributes["location"]
        self.check(scan, "R403", r'"lis" has no \'location\' attribute\.')

    def test_r403_conn_invalid_location(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        lis_var.attributes["location"] = "something"
        msg = (
            '"lis" has location="something", which is not one of '
            r'"face", "edge" or "node"\.'
        )
        self.check(scan, "R403", msg)

    def test_r404_conn_bad_location(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        lis_var.attributes["location"] = "edge"
        msg = (
            '"lis" has location="edge", which.*'
            r'does not exist in the parent mesh, "topology"\.'
        )
        self.check(scan, "R404", msg)

    def test_r405_conn_no_dims(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        # Set the lis to have no dimensions
        lis_var.dimensions = ()
        # Also remove the data-var, to avoid additional errors.
        del scan.variables["sample_data"]
        msg = r'"lis" has dimensions \(\), of which there are 0 instead of 1\.'
        self.check(scan, "R405", msg)

    def test_r405_conn_extra_dims(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        # Set the lis to have 2 dimensions
        lis_var.dimensions = ("lis_nodes", "num_ends")
        # Also remove the data-var, to avoid additional errors.
        del scan.variables["sample_data"]
        msg = (
            r"\"lis\" has dimensions \('lis_nodes', 'num_ends'\), of which "
            r"there are 2 instead of 1\."
        )
        self.check(scan, "R405", msg)

    def test_r406_conn_bad_start_index(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        # Set the lis to have an invalid start_index
        lis_var.attributes["start_index"] = 3
        msg = r'"lis" has start_index="3", which is not either 0 or 1\.'
        self.check(scan, "R406", msg)

    #
    # Advisory checks
    #

    def test_a401_conn_bad_dtype(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        lis_var.dtype = np.dtype(np.float32)
        msg = r'"lis" has type "float32", which is not an integer type\.'
        self.check(scan, "A401", msg)

    def test_a402_data_missing_points(self, scan_lis_w_data):
        scan = scan_lis_w_data
        lisvar = scan.variables["lis"]
        # Replace the data proxy to mimic different array contents
        array = lisvar.data.fetch_array()
        array = np.ma.masked_array(array)
        array[0] = np.ma.masked
        lisvar.data = FakeVariableDataWrapper(lisvar.data, array)
        # Note: not using a _FillValue, as that triggers another error
        msg = (
            'contains "missing" index values, which should not be '
            "present in a location-index-set"
        )
        self.check_withdata(scan, "A402", msg)

    def test_a403_conn_with_fillvalue(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        lis_var.attributes["_FillValue"] = -1
        msg = (
            "\"lis\" has a '_FillValue' attribute, which should not be "
            r"present on a location-index-set\."
        )
        self.check(scan, "A403", msg)

    def test_a404_conn_dim_longerthanparent(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        scan.dimensions["lis_nodes"] = NcDimSummary("lis_nodes", 17)
        msg = (
            '"lis" has dimension "lis_nodes", length 17.*'
            "longer than the node dimension of the parent "
            r'mesh "topology" : "num_node", length 8\.'
        )
        self.check(scan, "A404", msg)

    def test_a405_data_distinct_lis_values(self, scan_lis_w_data):
        scan = scan_lis_w_data
        lis_faces = scan.variables["lis"]
        lis_values = lis_faces.data.fetch_array()
        lis_values[:] = [1, 1]  # invalid due to repeating value
        lis_faces.data = FakeVariableDataWrapper(lis_faces.data, lis_values)
        msg = (
            "contains repeated index values, which should not be present "
            "in a location-index-set"
        )
        self.check_withdata(scan, "A405", msg)

    def test_a406_data_invalid_min(self, scan_lis_w_data):
        scan = scan_lis_w_data
        lis_faces = scan.variables["lis"]
        lis_values = lis_faces.data.fetch_array()
        lis_values[0] = 0  # invalid as start-index is 1
        lis_faces.data = FakeVariableDataWrapper(lis_faces.data, lis_values)
        msg = (
            "has some index value = 0, which is less than the "
            "start-index value of 1"
        )
        self.check_withdata(scan, "A406", msg)

    def test_a406_data_invalid_min_nostartindex(self, scan_lis_w_data):
        # As previous, but with default start-index
        scan = scan_lis_w_data
        lis_faces = scan.variables["lis"]
        del lis_faces.attributes["start_index"]
        lis_values = lis_faces.data.fetch_array()
        lis_values[:] = lis_values - 1  # adjust for start-index
        lis_values[1] = -3  # this is too small
        lis_faces.data = FakeVariableDataWrapper(lis_faces.data, lis_values)
        msg = (
            "has some index value = -3, which is less than the "
            "start-index value of 0"
        )
        self.check_withdata(scan, "A406", msg)

    def test_a406_data_invalid_max(self, scan_lis_w_data):
        scan = scan_lis_w_data
        lis_faces = scan.variables["lis"]
        lis_values = lis_faces.data.fetch_array()
        lis_values[0] = 4  # invalid
        lis_faces.data = FakeVariableDataWrapper(lis_faces.data, lis_values)
        msg = (
            "contains a maximum index value of 4, which is outside the range "
            r"of the relevant \"faces\" dimension, 1\.\.3\."
        )
        self.check_withdata(scan, "A406", msg)

    def test_a406_data_invalid_max_nostartindex(self, scan_lis_w_data):
        # As previous, but without a start-index attribute
        scan = scan_lis_w_data
        lis_faces = scan.variables["lis"]
        del lis_faces.attributes["start_index"]
        lis_values = lis_faces.data.fetch_array()
        lis_values = lis_values - 1
        lis_values[0] = 3  # invalid, as valid range is now 0..2
        lis_faces.data = FakeVariableDataWrapper(lis_faces.data, lis_values)
        msg = (
            "contains a maximum index value of 3, which is outside the range "
            r"of the relevant \"faces\" dimension, 0\.\.2\."
        )
        self.check_withdata(scan, "A406", msg)

    def test_a407_conn_start_index_badtype(self, scan_0d_and_lis):
        scan, lis_var = scan_0d_and_lis
        # Set the lis to have an invalid start_index
        lis_var.attributes["start_index"] = np.array(1, dtype=np.float16)
        msg = (
            '"lis" has a \'start_index\' of type "float16", '
            r'which is different from the variable type, "int64"\.'
        )
        self.check(scan, "A407", msg)


class TestChecker_Global(DatasetChecker):
    # A901 : not testable (except via cf-checker)

    def test_a902_conventions_missing(self, scan_0d_mesh):
        del scan_0d_mesh.attributes["Conventions"]
        msg = r"A902 : dataset has no 'Conventions' attribute\."
        self.check(scan_0d_mesh, "A902", msg)

    def test_a903_conventions_nougrid(self, scan_0d_mesh):
        scan_0d_mesh.attributes["Conventions"] = "something; CF-1.2"
        msg = (
            'A903 : dataset has Conventions="something; CF-1.2", which '
            r'does not.* of the form "UGRID-<major>.<minor>"\.'
        )
        self.check(scan_0d_mesh, "A903", msg)

    # A904 : not really testable, since assigning *any* valid cf_role prevents
    # a variable being identified a plain non-UGRID var (or even a datavar)
    # (though that could perhaps change in future).
    # For example...
    def test_a904_rogue_ugrid_cfrole(self, scan_0d_mesh):
        oddvar_name = "rogue"
        scan_0d_mesh.variables[oddvar_name] = NcVariableSummary(
            name=oddvar_name,
            dimensions=("num_node",),
            shape=(8,),
            dtype=np.dtype(np.int32),
            attributes={
                "cf_role": "face_face_connectivity",
            },
        )
        self.check(
            scan_0d_mesh,
            statements=[
                ("A301", r'connectivity variable "rogue".* no parent mesh\.'),
                ("R304", '"rogue" has dimensions.* 1, instead of 2.'),
            ],
        )

    def test_a905_invalid_cf_role(self, scan_0d_mesh):
        oddvar_name = "rogue"
        scan_0d_mesh.variables[oddvar_name] = NcVariableSummary(
            name=oddvar_name,
            dimensions=("num_node",),
            shape=(8,),
            dtype=np.dtype(np.int32),
            attributes={
                "cf_role": "something_odd",
            },
        )
        msg = (
            'WARN A905 : netcdf variable "rogue" has cf_role="something_odd", '
            "which is not a recognised cf-role value "
            "defined by either CF or UGRID."
        )
        self.check(scan_0d_mesh, "A905", msg)
