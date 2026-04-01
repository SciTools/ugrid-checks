from .. import simple_scan
from ... import cdl_scanner
from ..test_check_dataset__checks import (
    DatasetChecker,
    scan_0d_mesh,
    scan_1d_mesh,
    scan_2d_mesh,
)
from .test_structure_report import simple_scan_w_nonmesh

# Yes, we do need these imports.
cdl_scanner
simple_scan
scan_0d_mesh
scan_1d_mesh
scan_2d_mesh
simple_scan_w_nonmesh


from ugrid_checks.check import Checker


class Test_StructureObject(DatasetChecker):
    def get_struct(self, scan, include_nonmesh=False):
        checker = Checker(scan, max_mb_checks=0)
        struct = checker.structure_object()
        return struct

    def test_basic(self, simple_scan_w_nonmesh):
        scan = simple_scan_w_nonmesh
        result = self.get_struct(scan)
        print(result)

    def test_2d(self, scan_2d_mesh):
        scan = scan_2d_mesh
        result = self.get_struct(scan)
        print(result)
