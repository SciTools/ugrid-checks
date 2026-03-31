from ugrid_checks.check import Checker
from ugrid_checks.nc_dataset_scan import NcDimSummary, NcVariableSummary

from . import cdl_scanner
from .checker.test_structure_report import simple_scan, simple_scan_w_nonmesh

# Yes, we do need these imports
cdl_scanner
simple_scan
simple_scan_w_nonmesh


class TestStructureReporter:
    """Test the python API of the "structure" object."""

    def test_basic_mesh_and_nonmesh(self, simple_scan_w_nonmesh):
        # A single, very simple testcase, based on :
        # test_structure_report.Test_StructureReport.test_basic__w_nonmesh
        struct = Checker(simple_scan_w_nonmesh).file_structure()

        # Apply some basic checks to each of the structure properties.
        assert set(struct.all_file_dims.keys()) == {"num_node", "x_dim"}
        sample_dim = struct.all_file_dims["num_node"]
        assert isinstance(sample_dim, NcDimSummary)
        assert sample_dim == NcDimSummary(8)

        assert set(struct.all_file_vars.keys()) == {
            "topology",
            "sample_data",
            "node_lon",
            "x_data",
            "node_lat",
        }
        sample_var = struct.all_file_vars["x_data"]
        assert isinstance(sample_var, NcVariableSummary)
        assert sample_var.name == "x_data"

        assert list(struct.mesh_vars.keys()) == ["topology"]
        assert list(struct.lis_vars.keys()) == []
        assert list(struct.meshdata_vars.keys()) == ["sample_data"]
        assert list(struct.orphan_connectivities.keys()) == []

        assert struct.mesh_location_dims == {
            "topology": {
                "boundary": None,
                "node": "num_node",
                "edge": None,
                "face": None,
            }
        }
        assert list(struct.nonmesh_dims.keys()) == ["x_dim"]
        assert struct.nonmesh_dims["x_dim"] == NcDimSummary(3)

        assert list(struct.nonmesh_vars.keys()) == ["x_data"]
        assert isinstance(struct.nonmesh_vars["x_data"], NcVariableSummary)
