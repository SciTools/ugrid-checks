"""
Tests for :meth:`ugrid_checks.check.Checker.structure_report()`.

"""

import re

import numpy as np
from pytest import fixture
from ugrid_checks.nc_dataset_scan import NcDimSummary, NcVariableSummary

from .. import simple_scan
from ... import cdl_scanner
from ..test_check_dataset__checks import (
    DatasetChecker,
    scan_0d_mesh,
    scan_1d_mesh,
    scan_2d_mesh,
)

# Yes, we do need these imports.
cdl_scanner
simple_scan
scan_0d_mesh
scan_1d_mesh
scan_2d_mesh


from ugrid_checks.check import Checker


@fixture
def nomesh_scan(cdl_scanner):
    test_cdl = """
    netcdf data_C4 {
    dimensions:
        x = 8 ;

    variables:
        double var_1(x) ;
    }
    """
    scan = cdl_scanner.scan(test_cdl)
    return scan


@fixture
def simple_scan_w_nonmesh(simple_scan):
    # Add an extra non-mesh dim and variable, for nonmesh output testing.
    scan = simple_scan
    scan.dimensions["x_dim"] = NcDimSummary("x_dim", 3)
    scan.variables["x_data"] = NcVariableSummary(
        name="x_data", dimensions=("x_dim",), shape=(3,), dtype=np.int32
    )
    return scan


class Test_StructureReport(DatasetChecker):
    def get_report(self, scan, include_nonmesh=False):
        checker = Checker(scan, max_mb_checks=0)
        text = checker.structure_report(include_nonmesh=include_nonmesh)
        return text

    def expect_in_text(self, text, expects):
        """
        Check that a (list of) expected regular-expressions occur in a text,
        in the order given.

        If they aren't all present, produce an informative message showing
        which are missing.
        N.B. uses "assert list == []", for intelligible error messages.
        E.G.
            self.expect_in_text(['XXXX', '.*'], "this"") == []
            -->
                ['XXXX'] != []

                Expected :[]
                Actual   :['XXXX']

        """
        if isinstance(expects, str):
            expects = [expects]
        unmatched_expects = []
        for expect in expects:
            # Match each 'expect' entry to the result text
            match = re.search(expect, text, re.DOTALL)
            if match:
                # Item found.  Continue from after match
                text = text[match.end() :]
            else:
                # This not found.  Continue from same place
                unmatched_expects.append(expect)
        assert unmatched_expects == []

    def test_empty__nomesh__meshonly(self, nomesh_scan):
        result = self.get_report(nomesh_scan)
        expect = "Meshes : <none>"
        self.expect_in_text(result, expect)

    def test_empty__nomesh__w_nonmesh(self, nomesh_scan):
        # Do "include_nonmesh", on a no-mesh dataset.
        result = self.get_report(nomesh_scan, include_nonmesh=True)
        expects = [
            "Meshes : <none>",
            r'Non-mesh[^\n]*\n *dimensions:\n *"x"',
            r'variables:\n *"var_1"',
        ]
        self.expect_in_text(result, expects)

    def test_basic__meshonly(self, simple_scan_w_nonmesh):
        # Simple case : just for this, do a full text comparison.
        text = self.get_report(simple_scan_w_nonmesh)
        text_lines = text.split("\n")
        expected_lines = [
            "Meshes",
            '    "topology"',
            '        node("num_node")',
            '            coordinates : "node_lat", "node_lon"',
            "",
            "Mesh Data Variables",
            '    "sample_data"',
            '        mesh : "topology"',
            '        location : "node"',
        ]
        assert text_lines == expected_lines

    def test_basic__w_nonmesh(self, simple_scan_w_nonmesh):
        # Check the "nonmesh" output : another full-text check.
        text = self.get_report(simple_scan_w_nonmesh, include_nonmesh=True)
        text_lines = text.split("\n")
        expected_lines = [
            "Meshes",
            '    "topology"',
            '        node("num_node")',
            '            coordinates : "node_lat", "node_lon"',
            "",
            "Mesh Data Variables",
            '    "sample_data"',
            '        mesh : "topology"',
            '        location : "node"',
            "",
            "Non-mesh variables and/or dimensions",
            "    dimensions:",
            '        "x_dim"',
            "    variables:",
            '        "x_data"',
        ]
        assert text_lines == expected_lines

    def test_missing_node_coords(self, simple_scan):
        scan = simple_scan
        del scan.variables["topology"].attributes["node_coordinates"]
        text = self.get_report(scan)
        # text = self.get_report(scan)
        expects = [
            (
                'Meshes\n *"topology"\n *'
                r"\<\? no node coordinates or dimension \?\>"
            ),
        ]
        self.expect_in_text(text, expects)

    def test_multiple_locations(self, scan_2d_mesh):
        # Test with locations other than 'node'
        text = self.get_report(scan_2d_mesh)
        expects = [
            (
                r'  node\("num_node"\)'
                '\n *coordinates : "node_lat", "node_lon"\n'
            ),
            r'  face\("face_dim"\)\n',
            '  face_node_connectivity : "face_nodes"\n',
            '  coordinates : "latitude", "longitude"\n',
        ]
        self.expect_in_text(text, expects)

    def test_no_face_coords(self, scan_2d_mesh):
        # Test with node coords but no face coords
        del scan_2d_mesh.variables["topology"].attributes["face_coordinates"]
        text = self.get_report(scan_2d_mesh)
        assert 'coordinates : "latitude", "longitude"' not in text

    def test_lis(self, scan_0d_mesh):
        # Test with a location-index set
        scan = scan_0d_mesh
        self._convert_to_lis(scan)
        text = self.get_report(scan)
        expects = [
            "\nLocation Index Sets",
            r'  "lis" \(lis_nodes\)',
            '  mesh : "topology"',
            "  location : node",
        ]
        self.expect_in_text(text, expects)

    def test_optional_connectivities(self, scan_2d_mesh):
        scan = scan_2d_mesh
        self._add_edges(scan, with_facedge_conn=True)
        text = self.get_report(scan)
        expects = [
            "optional connectivities",
            'face_edge_connectivity : "face_edges_var"',
        ]
        self.expect_in_text(text, expects)

    def test_blank_connectivity(self, scan_2d_mesh):
        scan = scan_2d_mesh
        self._add_edges(scan)
        # Add an extra 'face_edge_connectitivity' attribute that does *not*
        # target a valid variable.
        meshvar = scan.variables["topology"]
        meshvar.attributes["face_edge_connectivity"] = "bad_feconn"
        # This should appear with a reservation
        text = self.get_report(scan)
        expects = [
            "optional connectivities",
            r'  face_edge_connectivity : \<\?nonexistent\?\> "bad_feconn"',
        ]
        self.expect_in_text(text, expects)

    def test_orphan_connectivities(self, scan_2d_mesh):
        scan = scan_2d_mesh
        self._add_edges(scan, with_facedge_conn=True)
        del scan.variables["topology"].attributes["face_edge_connectivity"]
        text = self.get_report(scan)
        expects = [
            ("\n" r"\?\? Connectivities with no mesh \?\?" "\n"),
            r'  "face_edges_var" \("face_n_edges", "num_vertices"\)',
            '  cf_role = "face_edge_connectivity"',
        ]
        self.expect_in_text(text, expects)

    def test_extra_datavar_prop(self, scan_2d_mesh):
        # Test output when a mesh datavar has an extra 'lis' attribute
        scan = scan_2d_mesh
        scan.variables["sample_data"].attributes["location_index_set"] = "junk"
        text = self.get_report(scan)
        expects = [
            "Mesh Data Variables",
            '  "sample_data"',
            '    location : "face"',
            r'    location_index_set : \<\?unexpected\?\> "junk"',
        ]
        self.expect_in_text(text, expects)

    def test_missing_datavar_prop(self, scan_2d_mesh):
        # Test output when a mesh datavar has no 'location' attribute
        scan = scan_2d_mesh
        del scan.variables["sample_data"].attributes["location"]
        text = self.get_report(scan)
        expects = [
            "Mesh Data Variables",
            '  "sample_data"',
            '    mesh : "topology"',
            r"    location : \<\?missing\?\>",
        ]
        self.expect_in_text(text, expects)

    def test_nonmesh_coordbounds(self, scan_1d_mesh):
        scan = scan_1d_mesh
        # Add an extra variable, suitable for node-longitude bounds
        scan.variables["lon_bounds"] = NcVariableSummary(
            name="lon_bounds",
            dimensions=["num_node", "num_ends"],
            shape=(8, 2),
            dtype=np.dtype(np.float32),
        )
        # At this point, it appears as a 'non-mesh var'
        text = self.get_report(scan, include_nonmesh=True)
        expects = [
            "Non-mesh variables and/or dimensions",
            "  variables:",
            '    "lon_bounds"',
        ]
        self.expect_in_text(text, expects)

        # Reference the new var as a mesh coord bounds
        scan.variables["node_lon"].attributes["bounds"] = "lon_bounds"
        text = self.get_report(scan, include_nonmesh=True)
        # Now *NO* "non-mesh" section, because it is considered a mesh var
        assert "Non-mesh" not in text

    def test_nonmesh_connectivity_dims(self, scan_2d_mesh):
        scan = scan_2d_mesh
        self._add_edges(scan)
        # Add an extra dimension
        scan.dimensions["face_n_edges"] = NcDimSummary("face_n_edges", 5)
        # This should appear in a non-mesh section
        text = self.get_report(scan, include_nonmesh=True)
        expects = [
            "Non-mesh variables and/or dimensions",
            "  dimensions:",
            '    "face_n_edges"',
        ]
        self.expect_in_text(text, expects)
        # No optional connectivitites
        assert "optional conn" not in text

        # Now add a face-edge connectivity, mapping this dimension
        self._add_faceedge_conn(scan)
        text = self.get_report(scan, include_nonmesh=True)
        expects = [
            "optional connectivities",
            '  face_edge_connectivity : "face_edges_var"',
        ]
        self.expect_in_text(text, expects)
        # Now *NO* "non-mesh" section, because it is considered a mesh var
        assert "Non-mesh" not in text

    def test_nonmesh_orphan_connectivity_dims(self, scan_0d_mesh):
        scan = scan_0d_mesh
        # Add an extra dim..
        scan.dimensions["extra_dim"] = NcDimSummary("extra_dim", 3)
        # .. which appears as a 'nonmesh' dim
        text = self.get_report(scan, include_nonmesh=True)
        expects = [
            "Non-mesh variables and/or dimensions",
            "  dimensions:",
            '    "extra_dim"',
        ]
        self.expect_in_text(text, expects)
        assert "Connectivities with no mesh" not in text

        # Add an orphan connectivity which maps this dim..
        scan.variables["extra_conn"] = NcVariableSummary(
            name="extra",
            dimensions=(
                "num_node",
                "extra_dim",
            ),
            shape=(3,),
            dtype=np.dtype(np.int64),
            attributes={"cf_role": "face_edge_connectivity"},
        )
        # .. which *prevents* the dim appearing as a non-mesh section
        text = self.get_report(scan, include_nonmesh=True)
        expects = [
            r"\?\? Connectivities with no mesh \?\?",
            r'  "extra_conn" \("num_node", "extra_dim"\)',
            '  cf_role = "face_edge_connectivity"',
        ]
        self.expect_in_text(text, expects)
        assert "Non-mesh" not in text
