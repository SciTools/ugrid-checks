from pathlib import Path
import re
from typing import AnyStr, Dict, List, Mapping, Text, Tuple, Union

import numpy as np

from ._var_data import VariableDataStats
from .nc_dataset_scan import (
    NcDimSummary,
    NcFileSummary,
    NcVariableSummary,
    scan_dataset,
)
from .scan_utils import (
    property_as_single_name,
    property_namelist,
    vars_w_props,
)
from .structure import UgridDataset, UgridDatavar, UgridLis, UgridMesh
from .ugrid_logger import CheckLoggingInterface

__all__ = ["Checker", "check_dataset"]

_VALID_UGRID_LOCATIONS = [
    "node",
    "edge",
    "face",  # Not supporting 'volume' at present
]

_VALID_CONNECTIVITY_ROLES = [
    "edge_node_connectivity",
    "face_node_connectivity",
    "face_edge_connectivity",
    "edge_face_connectivity",
    "face_face_connectivity",
    "boundary_node_connectivity",
]

_VALID_UGRID_CF_ROLES = [
    "mesh_topology",
    "location_index_set",
] + _VALID_CONNECTIVITY_ROLES

_VALID_MESHCOORD_ATTRS = [
    f"{location}_coordinates" for location in _VALID_UGRID_LOCATIONS
]

_VALID_CF_CF_ROLES = [
    "timeseries_id",
    "profile_id",
    "trajectory_id",
]

# Valid varname regex.
# It is not possible to use the common 'variable name' patterns,
# as we *can* legally have variables which start with a digit, etc.
# In principle, in netcdf anything but forward-slash is allowed
#   - c.f. https://github.com/cf-convention/cf-conventions/issues/307
# However, clearly, we can't really handle spaces in names.
# FOR NOW: allow a Python "normal word character", followed by anything except
# the forward-slash and space.
_VALID_NAME_REGEX = re.compile(r"^\w[^ /]*$")


class Checker:
    """
    Object to perform UGRID checking on a file.

    Scans a file on creation, and records the checking messages on its
    'self.logger', which is a :class:`CheckLoggingInterface`.

    Can produce text reports for a checking summary, and a file structure
    summary.

    Could also be used programmatically to aid file analysis, but the way the
    information is stored is not currently designed with external use in mind.

    """

    def __init__(
        self,
        file_scan: NcFileSummary,
        logger: CheckLoggingInterface = None,
        ignore_warnings=False,
        ignore_codes: Union[List[str], None] = None,
        max_mb_checks: float = -1.0,  # Check *any* data, by default.
    ):
        self.file_scan = file_scan
        if logger is None:
            logger = CheckLoggingInterface()
        self.logger = logger
        if max_mb_checks < 0:
            max_mb_checks = 1.0e30  # a ridiculously large number
        self.max_mb_checks = max_mb_checks
        if ignore_codes is None:
            ignore_codes = []
        self.ignore_codes = ignore_codes
        self.ignore_warnings = ignore_warnings
        # A shortcut for all the variables
        self._all_vars = file_scan.variables
        # Note: the following are filled in by 'dataset_identify_containers'
        self._meshdata_vars: Dict[str, NcVariableSummary] = {}
        self._mesh_vars: Dict[str, NcVariableSummary] = {}
        self._lis_vars: Dict[str, NcVariableSummary] = {}
        self._mesh_referrers: Dict[str, str] = {}
        self._lis_referrers: Dict[str, str] = {}
        # Note: these are filled by 'dataset_check_containers_and_map_dims'
        self._all_mesh_dims: Dict[str, Dict[str, Union[None, str]]] = {}
        self._allowed_cfrole_varnames: List[str]
        self._orphan_connectivities: Dict[str, NcVariableSummary] = {}
        self.data_skipped = False
        # Initialise
        self.check_dataset()

    def _data_skipped_event(self):
        # Register that, during data-checks, some data was too large.
        self.data_skipped = True

    def state(self, errcode: str, vartype: str, varname: str, msg: str):
        """
        Log a checking statement.

        Interface as for :meth:`CheckLoggingInterface.state`.

        """
        if errcode not in self.ignore_codes:
            if not self.ignore_warnings or not errcode.startswith("A"):
                self.logger.state(errcode, vartype, varname, msg)

    def check_mesh_attr_is_varlist(
        self, meshvar: NcVariableSummary, attrname: str
    ):
        """
        Check that a mesh-var attribute, if it exists, is a valid varlist.

        Parameters
        ----------
        meshvar : class:`NcVariableSummary`
            mesh variable
        attrname : str
            name of the attribute of 'meshvar' to check

        Returns
        -------
        ok : bool
            True iff no problems were found

        """
        value = meshvar.attributes.get(attrname)
        if value is None:
            # Missing is ok.  But NB *not* an empty string (see below).
            success = True
        else:
            success = value.dtype.kind == "U"
            if not success:
                msg = (
                    f"attribute '{attrname}' has type \"{value.dtype}\", "
                    "which is not a string type."
                )
                self.state("R105", "Mesh", meshvar.name, msg)
            if success:
                varnames = property_namelist(value)
                if not varnames:
                    # Empty is *not* a valid content.
                    # N.B. this includes non-string contents.
                    self.state(
                        "R105",
                        "Mesh",
                        meshvar.name,
                        f'has {attrname}="{value}", '
                        "which is not a valid list of variable names.",
                    )
                    success = False
            if success:
                for varname in varnames:
                    if not varname:  # skip any extra blanks
                        continue
                    if not _VALID_NAME_REGEX.match(varname):
                        self.state(
                            "R105",
                            "Mesh",
                            meshvar.name,
                            f'has {attrname}="{varname}", '
                            "which is not a valid variable name.",
                        )
                        success = False
                    elif varname not in self._all_vars:
                        self.state(
                            "R106",
                            "Mesh",
                            meshvar.name,
                            f"attribute '{attrname}' refers to a variable "
                            f'"{varname}", but there is no such variable '
                            "in the dataset.",
                        )
                        success = False

        return success

    def var_ref_problem(self, attr_value: np.ndarray) -> str:
        """
        Make a text description of any problems of a single-variable reference.

        Check that the input contains a single, valid name, referring to an
        existing variable.
        If no problem, returns an empty string.

        """
        succeed = True
        if attr_value.dtype.kind != "U":
            result = "is not a string value"
            succeed = False
        if succeed:
            names = property_namelist(attr_value)
            if len(names) != 1:
                result = "is not a single variable name"
                succeed = False
        if succeed:
            var_name = property_as_single_name(attr_value)
            if not _VALID_NAME_REGEX.match(var_name):
                result = "is not a valid variable name"
                succeed = False
        if succeed:
            bounds_var = self._all_vars.get(var_name)
            if bounds_var is None:
                result = "is not a variable in the dataset"
                succeed = False
        if succeed:
            result = ""
        return result

    def check_coord_bounds(
        self,
        coord: NcVariableSummary,
        meshvar: NcVariableSummary,
        location: str,
    ) -> List[Tuple[str]]:
        """
        Validity-check the bounds of a coordinate (if any).

        Ok for _no_ bounds-attribute, but not if it is an empty string.
        Check: existence, n-dims, parent dimension, standard-name and units.

        Note: this method does not log messages directly, but returns results
        for the caller to log them with added context.

        Returns
        codes_and_messages : List[tuple(str, str)]
            a list of codes and messages, to be logged in the context of the
            parent coordinate variable.

        """
        bounds_name = coord.attributes.get("bounds")
        result_codes_and_messages = []

        def log_bounds_statement(code, msg):
            msg = f'has bounds="{bounds_name}", which {msg}'
            result_codes_and_messages.append((code, msg))

        has_bounds = bounds_name is not None
        if has_bounds:
            msg = self.var_ref_problem(bounds_name)
            if msg != "":
                log_bounds_statement("R203", f"{msg}.")  # NB full stop !
                has_bounds = False

        if has_bounds:
            # NB from the above check, we do have a bounds variable.
            bounds_var = self._all_vars[str(bounds_name)]
            bounds_dims = bounds_var.dimensions
            (coord_dim,) = coord.dimensions  # NB always has exactly 1
            if coord_dim not in bounds_dims:
                msg = (
                    f"has dimensions {bounds_dims!r}, which does not include "
                    f'the parent variable dimension, "{coord_dim}".'
                )
                log_bounds_statement("R203", msg)

            n_bounds_dims = len(bounds_dims)
            if n_bounds_dims != 2:
                msg = (
                    f"has dimensions {bounds_dims!r}, of which there should "
                    f"be 2, instead of {n_bounds_dims}."
                )
                log_bounds_statement("R203", msg)

            #
            # Advisory checks
            #

            def check_attr_mismatch(attr_name):
                coord_attr, bounds_attr = [
                    var.attributes.get(attr_name)
                    for var in (coord, bounds_var)
                ]
                if bounds_attr is not None and bounds_attr != coord_attr:
                    if coord_attr is None:
                        coord_attr = "<none>"
                    msg = (
                        f'has {attr_name}="{bounds_attr}", which does not '
                        f"match the parent '{attr_name}' of \"{coord_attr}\"."
                    )
                    log_bounds_statement("R203", msg)

            check_attr_mismatch("standard_name")
            check_attr_mismatch("units")

            if location == "node":
                msg = (
                    "has a 'bounds' attribute, which is not valid for a "
                    "coordinate of location 'node'."
                )
                log_bounds_statement("A206", msg)
            else:
                # Do the data-values check.  This is potentially costly.
                # Check the bounds against those *calculated* from the relevant
                # connectivity + node coordinates.
                assert location in ("edge", "face")

                # Find the relevant (required) connectivity var
                connvar_name = property_as_single_name(
                    meshvar.attributes.get(f"{location}_node_connectivity")
                )
                connvar = self._all_vars.get(connvar_name)

                if connvar:
                    # Check the shapes match as expected.
                    if bounds_var.shape != connvar.shape:
                        msg = (
                            f"has shape {bounds_var.shape}, which does not "
                            f"match the shape {connvar.shape} of the "
                            f'{location} connectivity, "{connvar.name}".'
                        )
                        log_bounds_statement("A205", msg)
                        connvar = None  # Abandon the subsequent checks

                # Find a node coord with stdname + units matching this coord
                # NOTE: node_coord=None is used to abort further calculation
                node_coord = None
                coord_stdname = coord.attributes.get("standard_name")
                if connvar and coord_stdname:
                    # Find the matching node coordinate
                    # N.B. we need a std-name, but can tolerate missing units
                    coord_units = coord.attributes.get("units")
                    node_coord_names = property_namelist(
                        meshvar.attributes.get("node_coordinates")
                    )
                    for node_coord_name in node_coord_names:
                        var = self._all_vars.get(node_coord_name)
                        if not var:
                            continue
                        stdname = var.attributes.get("standard_name") or ""
                        if stdname != coord_stdname:
                            continue
                        if coord_units:
                            units = var.attributes.get("units") or ""
                            if units != coord_units:
                                continue
                        # ELSE: this is the one wanted
                        node_coord = var
                        break

                if node_coord:
                    # Fetch the connectivity and node-coord data arrays
                    size_lim = self.max_mb_checks * 0.33
                    conn_nodeinds = VariableDataStats(
                        connvar, size_lim, self._data_skipped_event
                    ).get_data()
                    if conn_nodeinds is not None:
                        # Adjust for start-index
                        start_index = connvar.attributes.get("start_index", 0)
                        conn_nodeinds -= start_index
                    node_coordvals = VariableDataStats(
                        node_coord, size_lim, self._data_skipped_event
                    ).get_data()
                    if conn_nodeinds is None or node_coordvals is None:
                        # disable if too costly (arrays did not load).
                        node_coord = None

                if node_coord:
                    # Construct calculated  bounds.
                    # A potentially costly 'fancy indexing' calculation
                    expected_coord_bounds = node_coordvals[conn_nodeinds]

                    # Mask the 'expected' array at points which are masked
                    # (missing) in the connectivity array, since the
                    # fancy-indexing op does *not* do that for us.
                    # The np.allclose below will then skip those points.
                    if np.ma.is_masked(conn_nodeinds):
                        expected_coord_bounds.mask |= conn_nodeinds.mask

                    # discard component arrays to reduce memory, maybe
                    conn_nodeinds = None
                    node_coordvals = None

                    # Fetch the bounds-variable data
                    bounds_data = VariableDataStats(
                        bounds_var, size_lim, self._data_skipped_event
                    ).get_data()
                    if bounds_data is None:
                        node_coord = None

                if node_coord:
                    # Compare bounds data against calculation
                    # - a potentially even-more-costly calculation (!)
                    if not np.allclose(expected_coord_bounds, bounds_data):
                        msg = (
                            f"does not match the expected values "
                            f'calculated from the "{node_coord.name}" '
                            f"node coordinate and the {location} "
                            f'connectivity, "{connvar.name}".'
                        )
                        log_bounds_statement("A205", msg)

        return result_codes_and_messages

    def check_coordinate(
        self,
        coord: NcVariableSummary,
        meshvar: NcVariableSummary,
        location: str,
    ):
        """Validity-check a coordinate of a mesh."""
        common_msg_prefix = f"within {meshvar.name}:{location}_coordinates "

        # Function to emit a statement message, adding context as to the
        # specific coord variable.
        def log_coord(code, msg):
            self.state(
                code, "Mesh coordinate", coord.name, common_msg_prefix + msg
            )

        coord_ndims = len(coord.dimensions)
        if coord_ndims != 1:
            msg = (
                f"should have exactly one dimension, but has "
                f"{coord_ndims} dimensions : {coord.dimensions!r}."
            )
            log_coord("R201", msg)
        else:
            # Check the dimension is the correct one according to location.
            (coord_dim,) = coord.dimensions
            mesh_dim = self._all_mesh_dims[meshvar.name][location]
            if coord_dim != mesh_dim:
                msg = (
                    f'has dimension "{coord_dim}", but the parent mesh '
                    f'{location} dimension is "{mesh_dim}".'
                )
                log_coord("R202", msg)
            # Check coord bounds (if any)
            # N.B. this *also* assumes a single dim for the primary var
            codes_and_messages = self.check_coord_bounds(
                coord, meshvar, location
            )
            for code, msg in codes_and_messages:
                log_coord(code, msg)

        #
        # Advisory notes..
        #

        # A201 should have 1-and-only-1 parent mesh : this is handled by
        # 'check_dataset', as it involves multiple meshes.

        # A202 floating-point type
        dtype = coord.dtype
        if dtype.kind != "f":
            log_coord(
                "A202",
                f'has type "{dtype}", which is not a floating-point type.',
            )

        # A203 standard-name : has+valid (can't handle fully ??)
        stdname = coord.attributes.get("standard_name")
        if not stdname:
            log_coord("A203", "has no 'standard_name' attribute.")

        # A204 units : has+valid (can't handle fully ??)
        stdname = coord.attributes.get("units")
        if not stdname:
            log_coord("A204", "has no 'units' attribute.")

        # A205 bounds data values match derived ones
        # - did this already above, within "check_coord_bounds"

    def check_connectivity(
        self,
        conn_var: NcVariableSummary,
        meshvar: Union[NcVariableSummary, None] = None,
        role_name: Union[str, None] = None,
    ):
        """
        Validity-check a connectivity variable.

        This is either in the context of a containing 'meshvar', **or** with
        no containing mesh (so-called "orphan connectivity").
        In the 'orphan' case, both meshvar and role_name are None.

        """
        # Add to our list of variables 'allowed' to have a UGRID cf-role.
        conn_name = conn_var.name
        self._allowed_cfrole_varnames.append(conn_name)

        # Create a handler object for any data checks.
        conn_stats = VariableDataStats(
            conn_var, self.max_mb_checks, self._data_skipped_event
        )

        if meshvar:
            msg_prefix = f'of mesh "{meshvar.name}" '
        else:
            msg_prefix = ""

        def log_conn(errcode, msg):
            self.state(
                errcode, "Mesh connectivity", conn_name, msg_prefix + msg
            )

        cf_role = conn_var.attributes.get("cf_role")
        if cf_role is None:
            log_conn("R301", "has no 'cf_role' attribute.")
        elif cf_role not in _VALID_CONNECTIVITY_ROLES:
            msg = (
                f'has cf_role="{cf_role}", '
                "which is not a valid UGRID connectivity attribute."
            )
            log_conn("R302", msg)
        elif role_name and cf_role != role_name:
            msg = (
                f'has cf_role="{cf_role}", which is different from its '
                f'role in the parent mesh, which is "{role_name}".'
            )
            log_conn("R303", msg)

        if meshvar:
            # In the context of a meshvar, take 'role_name' as the definition.
            # -- we will then check the 'cf_role' attribute against that.
            assert role_name
        else:
            # With no meshvar, use the 'cf_role' attribute as our role
            # definition -- if there is one.
            role_name = str(cf_role) if cf_role else None

        conn_dims = conn_var.dimensions
        dims_msg = f"has dimensions {conn_dims!r}"
        if len(conn_dims) != 2:
            msg = (
                f"{dims_msg}, of which there are "
                f"{len(conn_dims)}, instead of 2."
            )
            log_conn("R304", msg)

        # A handy common term
        edgelike_conns = (
            "edge_node_connectivity",
            "boundary_node_connectivity",
        )

        if meshvar:
            # Check dims : can only be checked against a parent mesh
            mesh_dims = self._all_mesh_dims[meshvar.name]
            is_parent_dim = [dim in mesh_dims.values() for dim in conn_dims]
            n_parent_dims = sum(is_parent_dim)
            if n_parent_dims == 0:
                msg = (
                    f"{dims_msg}, which does not contain any element "
                    f"dimension of the parent mesh."
                )
                log_conn("R305", msg)
            elif n_parent_dims == len(conn_dims):
                msg = (
                    f"{dims_msg}, which does not contain any dimension "
                    f"which is not an element dimension of the parent mesh."
                )
                log_conn("R306", msg)
            else:
                # Some are parent mesh-dims, and some not.
                # Just check that the *expected* mesh-dim is there.
                location = role_name.split("_")[0]
                parent_dim = mesh_dims[location]
                if parent_dim not in conn_dims:
                    msg = (
                        f"{dims_msg}, which does not include the expected "
                        f"{location} dimension of the parent mesh, "
                        f'"{parent_dim}".'
                    )
                    log_conn("R307", msg)

            if role_name in edgelike_conns and n_parent_dims == 1:
                (conn_nonmesh_dim,) = (
                    dim
                    for dim, in_parent in zip(conn_dims, is_parent_dim)
                    if not in_parent
                )
                nonmesh_dim = self.file_scan.dimensions[conn_nonmesh_dim]
                nonmesh_length = nonmesh_dim.length
                if nonmesh_length != 2:
                    msg = (
                        f"{dims_msg}, which contains the non-mesh "
                        f'dimension "{conn_nonmesh_dim}", but this has '
                        f"length {nonmesh_length} instead of 2."
                    )
                    log_conn("R308", msg)

        index_offset = 0
        index_value = conn_var.attributes.get("start_index")
        if index_value is not None:
            # Note: check value, converted to int.
            # This avoids an extra warning for strings like "0", "1",
            # since a non-integral type triggers an A302 warning anyway.
            if int(index_value) not in (0, 1):
                msg = (
                    f'has start_index="{index_value}", which is not '
                    "either 0 or 1."
                )
                log_conn("R309", msg)
            else:
                index_offset = index_value

        if role_name in edgelike_conns:
            # Edge- and Boundary- node arrays may not have missing points.
            if conn_stats.has_missing_values:
                msg = (
                    "contains missing indices, which is not permitted for "
                    f'a connectivity of type "{role_name}".'
                )
                log_conn("R310", msg)
        elif role_name == "face_node_connectivity":
            # The face-node connectivity *can* have missing points,
            # but must have >= 3 valid nodes for each face
            if conn_stats.has_missing_values:
                data = conn_stats.get_data()
                if data is not None:
                    face_dimname = meshvar.attributes.get("face_dimension")
                    if face_dimname:
                        face_axis = conn_var.dimensions.index(face_dimname)
                    else:
                        face_axis = 0
                    vertex_axis = 1 - face_axis
                    mask = np.ma.getmaskarray(data)
                    face_node_counts = np.count_nonzero(
                        ~mask, axis=vertex_axis
                    )
                    if np.min(face_node_counts) < 3:
                        msg = "contains faces with less than 3 vertices."
                        log_conn("R311", msg)

        #
        # Advisory checks
        #

        # A301 1-and-only-1 parent mesh
        # In 'dataset_detect_multiple_refs', since it involves multiple meshes

        if conn_var.dtype.kind != "i":
            msg = (
                f'has type "{conn_var.dtype}", '
                "which is not an integer type."
            )
            log_conn("A302", msg)

        if index_value is not None and index_value.dtype != conn_var.dtype:
            msg = (
                f"has a 'start_index' of type \"{index_value.dtype}\", "
                "which is different from the variable type, "
                f'"{conn_var.dtype}".'
            )
            log_conn("A303", msg)

        fill_value = conn_var.attributes.get("_FillValue")
        if fill_value is None:
            # No fill value : check there are *no* missing indices.
            if conn_stats.has_missing_values:
                msg = (
                    "contains missing indices, "
                    "but has no '_FillValue' attribute."
                )
                log_conn("A305", msg)
        else:
            if role_name and role_name in (
                "boundary_node_connectivity",
                "edge_node_connectivity",
            ):
                msg = (
                    f"has a '_FillValue' attribute, which should not be present "
                    f'on a "{role_name}" connectivity.'
                )
                log_conn("A304", msg)

            if fill_value.dtype != conn_var.dtype:
                msg = (
                    f"has a '_FillValue' of type \"{fill_value.dtype}\", "
                    "which is different from the variable type, "
                    f'"{conn_var.dtype}".'
                )
                log_conn("A306", msg)

            if fill_value >= 0:
                msg = f'has _FillValue="{fill_value}", which is not negative.'
                log_conn("A307", msg)

        min_ind = conn_stats.min_value
        if min_ind < index_offset:
            msg = (
                f"contains a minimum index value of {min_ind}, "
                f"which is less than its start-index of {index_offset}."
            )
            log_conn("A308", msg)

        if meshvar:
            location = role_name.split("_")[1]  # the 'target' role
            parent_dim = mesh_dims[location]
            dim = self.file_scan.dimensions.get(parent_dim)
            if dim:
                max_ind = conn_stats.max_value
                if (max_ind - index_offset) >= dim.length:
                    msg = (
                        f"contains a maximum index value of {max_ind}, "
                        "which is outside the range of the relevant "
                        f'"{parent_dim}" dimension, '
                        f"{index_offset}..{dim.length + index_offset - 1}."
                    )
                    log_conn("A308", msg)

    def check_mesh_connectivity(
        self,
        meshvar: NcVariableSummary,
        attr_name: str,
    ):
        """Validity-check a connectivity attribute of a mesh-variable."""
        attr_value = meshvar.attributes.get(attr_name)
        ok = attr_value is not None
        if ok:
            conn_name = property_as_single_name(attr_value)
            ok = conn_name is not None
        if ok:
            conn_var = self._all_vars.get(conn_name)
            ok = conn_var is not None
        if ok:
            # Remove from the orphan list
            self._orphan_connectivities.pop(conn_name, None)
            # Check it, in the context of the containing mesh
            self.check_connectivity(conn_var, meshvar, attr_name)

    def check_mesh_var(self, meshvar: NcVariableSummary) -> Dict[str, str]:
        """
        Validity-check a mesh variable.

        Parameters
        ----------
        meshvar : :class:`NcVariableSummary`
            meshvar to check

        """

        def log_meshvar(code, msg):
            self.state(code, "Mesh", meshvar.name, msg)

        # First check for bad 'cf_role' :
        # if wrong, meshvar can only have been identified by reference.
        cfrole_prop = meshvar.attributes.get("cf_role", None)
        if cfrole_prop != "mesh_topology":
            # This variable does not have the expected 'cf_role', so if we are
            # checking it, it must be referred to as 'mesh' by some variable.
            referring_var_name = self._mesh_referrers[meshvar.name]
            # Either there is no 'cf_role', or it is "wrong".
            msg = (
                f"appears to be a mesh, "
                f'since it is the value of "{referring_var_name}:mesh". '
                "But it has "
            )
            if cfrole_prop is None:
                msg += "no 'cf_role' property,"
                errcode = "R101"
            else:
                msg += f'cf_role="{cfrole_prop}",'
                errcode = "R102"
            msg += ' which should be "mesh_topology".'
            # N.B. do not identify as a Mesh, statement just says "variable"
            self.state(errcode, "", meshvar.name, msg)
            # Also, if the 'cf_role' was something else, then check it is a
            # valid option + emit an additional message if needed.
            if (
                cfrole_prop is not None
                and cfrole_prop not in _VALID_UGRID_CF_ROLES
            ):
                msg = (
                    f'has cf_role="{cfrole_prop}", '
                    "which is not a valid UGRID cf_role."
                )
                log_meshvar("A905", msg)

        topology_dimension = meshvar.attributes.get("topology_dimension")
        if topology_dimension is None:
            log_meshvar("R103", "has no 'topology_dimension' attribute.")
        else:
            # Check the topology dimension.
            # In principle, this controls which other connectivity properties
            # may appear : In practice, it is better to parse those
            # independently, and then cross-check.
            if topology_dimension not in (0, 1, 2):
                msg = (
                    f'has topology_dimension="{topology_dimension}", '
                    "which is not 0, 1 or 2."
                )
                log_meshvar("R104", msg)
                # Handle this subsequently as if it was missing
                topology_dimension = None

        # Work out what topology-dimension is implied by the available mesh
        # properties, which we will use *instead* of the declared one in
        # subsequent tests (and check the declared one against it).
        highest_connectivity = None
        appropriate_dim = 0
        if "face_node_connectivity" in meshvar.attributes:
            appropriate_dim = 2
            highest_connectivity = "face_node_connectivity"
        elif "edge_node_connectivity" in meshvar.attributes:
            appropriate_dim = 1
            highest_connectivity = "edge_node_connectivity"

        if topology_dimension is not None:
            # Emit an error if the attributes present don't match the stated
            # topology-dimension.  If *no* topology-dimension, skip this : we
            # already flagged that it was missing, above.
            if topology_dimension != appropriate_dim:
                if topology_dimension == 0:
                    if appropriate_dim == 1:
                        errcode = "R111"  # unexpected edge-node
                    else:
                        assert appropriate_dim == 2
                        errcode = "R113"  # unexpected face-node
                elif topology_dimension == 1:
                    if appropriate_dim == 0:
                        errcode = "R112"  # missing edge-node
                    else:
                        assert appropriate_dim == 2
                        errcode = "R113"  # unexpected face-node
                else:
                    assert topology_dimension == 2
                    errcode = "R113"  # missing face-node

                if topology_dimension < appropriate_dim:
                    # something is extra
                    msg = (
                        f'has topology_dimension="{topology_dimension}", '
                        f"but the presence of a '{highest_connectivity}' "
                        f"attribute implies it should be {appropriate_dim}."
                    )
                else:
                    # something is missing
                    topology_required_attribute = {
                        0: "face_node",
                        1: "edge_node_connectivity",
                        2: "face_node_connectivity",
                    }[int(topology_dimension)]
                    msg = (
                        f'has topology_dimension="{topology_dimension}", '
                        f"but it has no '{topology_required_attribute}' "
                        f"attribute."
                    )
                log_meshvar(errcode, msg)

        # Check all coordinate and connectivity attributes are valid "varlists"
        varlist_names = _VALID_MESHCOORD_ATTRS + _VALID_CONNECTIVITY_ROLES
        for attr in varlist_names:
            is_conn = attr in _VALID_CONNECTIVITY_ROLES
            attr_value = meshvar.attributes.get(attr)
            if attr_value is not None:
                ok = self.check_mesh_attr_is_varlist(meshvar, attr)
                var_names = property_namelist(attr_value)
                if not ok:
                    errcode = "R109" if is_conn else "R108"
                    msg = (
                        f'has {attr}="{attr_value}", which is not '
                        "a list of variables in the dataset."
                    )
                    log_meshvar(errcode, msg)
                elif is_conn and len(var_names) != 1:
                    msg = (
                        f'has {attr}="{attr_value}", which contains '
                        f"{len(var_names)} names, instead of 1."
                    )
                    log_meshvar("R107", msg)

        # Work out the actual mesh dimensions.
        mesh_dims = {
            name: None for name in ("face", "edge", "node", "boundary")
        }
        self._all_mesh_dims[meshvar.name] = mesh_dims

        if "node_coordinates" not in meshvar.attributes:
            log_meshvar(
                "R110", "does not have a 'node_coordinates' attribute."
            )
        else:
            # Note: if a 'node_coordinates' attribute exists, then we already
            # checked that it is a valid varlist.
            # So don't re-raise any problems here, just press on.
            coord_names = property_namelist(
                meshvar.attributes["node_coordinates"]
            )
            if coord_names:
                coord_var = self._all_vars.get(coord_names[0])
                if coord_var:
                    # Answer is the first dimension, if any.
                    if len(coord_var.dimensions) > 0:
                        mesh_dims["node"] = coord_var.dimensions[0]

        def deduce_element_dim(location):
            # Identify the dim, and check consistency of relevant attributes.
            # If found, set it in 'mesh_dims'
            dimattr_name = f"{location}_dimension"
            connattr_name = f"{location}_node_connectivity"
            dimension_name = property_as_single_name(
                meshvar.attributes.get(dimattr_name)
            )
            if location in ("boundary", "node"):
                # No 'boundary_dimension' attribute is supported.
                if dimension_name:
                    dimension_name = None
                    msg = (
                        f"has an attribute '{dimattr_name}', which is not "
                        "a valid UGRID term, and may be a mistake."
                    )
                    log_meshvar("A106", msg)

            if dimension_name:
                # There is an explicit 'xxx_dimension' property.
                if connattr_name not in meshvar.attributes:
                    errcode = {"edge": "R123", "face": "R122"}[location]
                    msg = (
                        f"has an attribute '{dimattr_name}', "
                        "which is not valid "
                        f"since there is no '{connattr_name}'."
                    )
                    log_meshvar(errcode, msg)
                elif dimension_name in self.file_scan.dimensions:
                    mesh_dims[location] = dimension_name
                else:
                    errcode = {"edge": "R115", "face": "R117"}[location]
                    msg = (
                        f'has {dimattr_name}="{dimension_name}", which is not '
                        "a dimension in the dataset."
                    )
                    log_meshvar(errcode, msg)

            elif connattr_name in meshvar.attributes:
                # No "xxx_dimension" attribute, but we *do* have
                # "xxx_node_connectivity", so mesh does _have_ this location.
                connvar_name = property_as_single_name(
                    meshvar.attributes[connattr_name]
                )
                conn_var = self._all_vars.get(connvar_name)
                if conn_var:
                    # Answer is the first dimension, if any.
                    if len(conn_var.dimensions) > 0:
                        mesh_dims[location] = conn_var.dimensions[0]

        deduce_element_dim("node")
        deduce_element_dim("boundary")
        deduce_element_dim("edge")
        deduce_element_dim("face")

        # Check that, if any connectivities have non-standard dim order, then a
        # dimension attribute exists.
        def var_has_nonfirst_dim(varname, dimname):
            conn_var = self._all_vars.get(varname)
            result = conn_var is not None
            if result:
                result = dimname in conn_var.dimensions
            if result:
                result = conn_var.dimensions[0] != dimname
            return result

        location_altordered_conns = {}
        for attr in _VALID_CONNECTIVITY_ROLES:
            maindim_location = attr.split("_")[0]
            assert maindim_location != "node"  # no such connectivities
            maindim_name = mesh_dims[maindim_location]
            for conn_name in property_namelist(meshvar.attributes.get(attr)):
                if var_has_nonfirst_dim(conn_name, maindim_name):
                    # We found a connectivity with a nonstandard dim order
                    dim_attr = f"{maindim_location}_dimension"
                    if dim_attr not in meshvar.attributes:
                        # There is no corresponding 'xxx_dimension', so warn.
                        conns = location_altordered_conns.get(
                            maindim_location, set()
                        )
                        conns.add(conn_name)
                        location_altordered_conns[maindim_location] = conns

        for location, conns in location_altordered_conns.items():
            # Found connectivities with a nonstandard dim order for this dim.
            assert location in ("face", "edge")
            errcode = {"edge": "R116", "face": "R118"}[location]
            conn_names = [f'"{name}"' for name in conns]
            conn_names_str = ", ".join(conn_names)
            msg = (
                f"has no '{dim_attr}' attribute, but there are "
                f"{location} connectivities "
                f"with non-standard dimension order : {conn_names_str}."
            )
            log_meshvar(errcode, msg)

        # Check that all existing coordinates are valid.
        for coords_attr in _VALID_MESHCOORD_ATTRS:
            location = coords_attr.split("_")[0]
            # Only check coords of locations present in the mesh.
            # This avoids complaints about coords dis-connected by problems
            # with the topology identification.
            if mesh_dims[location]:
                coord_names = property_namelist(
                    meshvar.attributes.get(coords_attr)
                )
                for coord_name in coord_names:
                    coord = self._all_vars.get(coord_name)
                    if coord:
                        # If absent, that was already detected + logged.
                        self.check_coordinate(coord, meshvar, location)

        # Check that all existing connectivities are valid.
        for attr in _VALID_CONNECTIVITY_ROLES:
            self.check_mesh_connectivity(meshvar, attr)

        # deal with the optional elements (connectivities)
        def check_requires(errcode, attrname, location_1, location_2=None):
            exist = attrname in meshvar.attributes
            if exist:
                elems = [location_1]
                if location_2:
                    elems.append(location_2)
                required_elements = [
                    f"{name}_node_connectivity" for name in elems
                ]
                missing_elements = [
                    f"'{name}'"
                    for name in required_elements
                    if name not in meshvar.attributes
                ]
                if missing_elements:
                    err_msg = (
                        f"has a '{attrname}' attribute, which is not valid "
                        f"since there is no "
                    )
                    err_msg += "or ".join(missing_elements)
                    err_msg += " attribute present."
                    log_meshvar(errcode, err_msg)

        check_requires("R114", "boundary_node_connectivity", "face")
        check_requires("R119", "face_face_connectivity", "face")
        check_requires("R120", "face_edge_connectivity", "face", "edge")
        check_requires("R121", "edge_face_connectivity", "face", "edge")

        # Advisory checks.
        if meshvar.dimensions:
            log_meshvar("A101", "has dimensions.")
        if "standard_name" in meshvar.attributes:
            log_meshvar("A102", "has a 'standard_name' attribute.")
        if "units" in meshvar.attributes:
            log_meshvar("A103", "has a 'units' attribute.")
        # NOTE: "A104" relates to multiple meshvars, so is handled in caller.

        return mesh_dims

    def check_meshdata_var(self, datavar: NcVariableSummary):
        """Validity-check a mesh data variable."""

        def log_meshdata(errcode, msg):
            self.state(errcode, "Mesh data", datavar.name, msg)

        lis_name = datavar.attributes.get("location_index_set")
        mesh_name = datavar.attributes.get("mesh")
        location = datavar.attributes.get("location")
        # At least one of these is true, or we would not have identified this
        # as a mesh-data var.
        assert mesh_name is not None or lis_name is not None

        # Decide whether to check this as a lis-datavar or a mesh-datavar
        # This is designed to produce 3 possible "clash" errors:
        #   lis & mesh & ~location --> R506
        #   lis & location & ~mesh --> R507
        #   mesh & lis --> R501
        treat_as_lis = lis_name is not None and (
            mesh_name is None or location is None
        )

        # Initialise reference used for the generic parent dimension check
        parent_varname = None  # Can be either a meshvar or a lis
        parent_location = None
        if treat_as_lis:
            # Treat the datavar as a 'lis-datavar'
            #  --> has "location_index_set", but no "mesh" or "location"
            ref_msg = self.var_ref_problem(lis_name)
            if ref_msg:
                # Invalid 'location_index_set' reference
                msg = f'has location_index_set="{lis_name}", which {ref_msg}.'
                log_meshdata("R508", msg)
            else:
                # We have a valid lis var.  Take this as the 'parent' for
                # the generic dimension test R510
                parent_varname = str(lis_name)
                lis_var = self._lis_vars[parent_varname]
                # Also set the parent-location.
                # NOTE: we are not checking the lis-var here, only the datavar,
                # so just get a value that works if the lis is valid.
                parent_location = str(lis_var.attributes.get("location", ""))
                if parent_location not in _VALID_UGRID_LOCATIONS:
                    parent_location = None

            if mesh_name is not None:
                msg = (
                    "has a 'mesh' attribute, which is invalid since it is "
                    "based on a 'location_index_set' attribute."
                )
                log_meshdata("R506", msg)

            if location is not None:
                msg = (
                    "has a 'location' attribute, which is invalid since it is "
                    "based on a 'location_index_set' attribute."
                )
                log_meshdata("R507", msg)

        else:
            # Treat the datavar as a 'mesh-datavar'
            #  --> has "mesh" and "location", but no "location_index_set"
            ref_msg = self.var_ref_problem(mesh_name)
            if ref_msg:
                # Invalid 'mesh' reference
                msg = f'has mesh="{mesh_name}", which {ref_msg}.'
                log_meshdata("R502", msg)

            if lis_name is not None:
                msg = (
                    "has a 'location_index_set' attribute, which is invalid "
                    "since it is based on a 'mesh' attribute."
                )
                log_meshdata("R501", msg)

            if location is None:
                log_meshdata("R503", "has no 'location' attribute.")
            elif str(location) not in _VALID_UGRID_LOCATIONS:
                msg = (
                    f'has location="{location}", which is not one of '
                    f'"face", "edge" or "node".'
                )
                log_meshdata("R504", msg)
            else:
                # Given a valid location, check that it exists in the parent
                if not ref_msg:
                    parent_varname = str(mesh_name)
                    parent_location = str(location)
                    assert parent_varname in self._all_mesh_dims
                    mesh_dims = self._all_mesh_dims[parent_varname]
                    parent_dim = mesh_dims.get(parent_location)
                    if parent_dim is None:
                        msg = (
                            f'has location="{location}", which is a location '
                            "that does not exist in the parent mesh, "
                            f'"{parent_varname}".'
                        )
                        log_meshdata("R505", msg)

        # Generic dimension testing, for either lis- or mesh-type datavars
        # First check there is only 1 mesh-dim
        data_dims = datavar.dimensions
        data_mesh_dims = [
            dim
            for dim in data_dims
            if any(
                dim in self._all_mesh_dims[some_mesh_name].values()
                for some_mesh_name in self._all_mesh_dims
            )
        ]
        n_data_mesh_dims = len(data_mesh_dims)
        if n_data_mesh_dims != 1:
            msg = (
                f"has dimensions {data_dims}, of which {n_data_mesh_dims} "
                "are mesh dimensions, instead of 1."
            )
            log_meshdata("R509", msg)
            data_meshdim = None  # cannot check against parent
        else:
            # We have a single element-dim : check against a parent mesh or lis
            (data_meshdim,) = data_mesh_dims

        if parent_varname and parent_location and data_meshdim:
            # If we have a valid parent ref, and single mesh dimension of the
            # datavar, check that they match
            mesh_dims = self._all_mesh_dims[parent_varname]
            parent_dim = mesh_dims[parent_location]
            if parent_dim is not None and data_meshdim != parent_dim:
                # Warn only if the parent_dim *exists*, but does not match
                # N.B. missing parent dim is checked elsewhere : R505 or R404
                if parent_varname in self._lis_vars:
                    typename = "location_index_set"
                else:
                    typename = "mesh"
                msg = (
                    f'has the element dimension "{data_meshdim}", which does '
                    f"not match the {parent_location} dimension of the "
                    f'"{parent_varname}" {typename}, which is "{parent_dim}".'
                )
                log_meshdata("R510", msg)

    def check_lis_var(self, lis_var: NcVariableSummary):
        """Validity-check a location-index-set variable."""

        # Add the lis element dimension into self._all_mesh_dims
        dims = lis_var.dimensions
        if len(dims) == 1:
            # Lis has a valid location and single dim
            # So we can record 'our' dim as an element-dim
            (lis_dim,) = dims
            # Note: record this under **all** locations.
            # Since we want to recognise this as a 'mesh dim', even if the lis
            # has an invalid mesh or location, and we don't use this to check
            # it against the parent element dim.
            self._all_mesh_dims[lis_var.name] = {
                name: lis_dim for name in _VALID_UGRID_LOCATIONS
            }

        def log_lis(errcode, msg):
            self.state(errcode, "location-index-set", lis_var.name, msg)

        lisvar_stats = VariableDataStats(
            lis_var, self.max_mb_checks, self._data_skipped_event
        )

        cf_role = lis_var.attributes.get("cf_role")
        if cf_role is None:
            log_lis("R401", "has no 'cf_role' attribute.")
        elif cf_role != "location_index_set":
            msg = f'has cf_role="{cf_role}", instead of "location_index_set".'
            log_lis("R401", msg)

        mesh_var = None  # Used to skip additional checks when mesh is bad
        mesh_name = lis_var.attributes.get("mesh")
        if mesh_name is None:
            log_lis("R402", "has no 'mesh' attribute.")
        else:
            msg_ref = self.var_ref_problem(mesh_name)
            if msg_ref:
                # Some problem
                msg = f'has mesh="{mesh_name}", which {msg_ref}.'
                log_lis("R402", msg)
            else:
                # We have a valid mesh
                mesh_name = str(mesh_name)
                mesh_var = self._mesh_vars[mesh_name]

        location = lis_var.attributes.get("location")
        parent_dim = None
        if location is None:
            log_lis("R403", "has no 'location' attribute.")
        elif str(location) not in _VALID_UGRID_LOCATIONS:
            msg = (
                f'has location="{location}", which is not one of '
                '"face", "edge" or "node".'
            )
            log_lis("R403", msg)
        elif mesh_var:
            # check the location exists in the parent mesh
            location = str(location)
            mesh_dims = self._all_mesh_dims[mesh_name]
            parent_dim = mesh_dims[location]
            if parent_dim is None:
                msg = (
                    f'has location="{location}", which is a location '
                    "that does not exist in the parent mesh, "
                    f'"{mesh_name}".'
                )
                log_lis("R404", msg)
                # Don't attempt any further checks against the mesh
                mesh_var = None

        lis_dims = lis_var.dimensions
        n_lis_dims = len(lis_dims)
        if n_lis_dims != 1:
            msg = (
                f"has dimensions {lis_dims!r}, of which there are "
                f"{n_lis_dims} instead of 1."
            )
            log_lis("R405", msg)
            lis_dim = None
        else:
            (lis_dim,) = lis_dims

        index_offset = 0
        index_value = lis_var.attributes.get("start_index")
        if index_value is not None:
            # Note: check value, converted to int.
            # This avoids an extra warning for strings like "0", "1",
            # since a non-integral type triggers an A407 warning anyway.
            if int(index_value) not in (0, 1):
                msg = (
                    f'has start_index="{index_value}", which is not '
                    "either 0 or 1."
                )
                log_lis("R406", msg)
            else:
                index_offset = index_value

            if index_value.dtype != lis_var.dtype:
                msg = (
                    f"has a 'start_index' of type \"{index_value.dtype}\", "
                    "which is different from the variable type, "
                    f'"{lis_var.dtype}".'
                )
                log_lis("A407", msg)

        #
        # Advisory checks
        #
        if lis_var.dtype.kind != "i":
            msg = f'has type "{lis_var.dtype}", which is not an integer type.'
            log_lis("A401", msg)

        if lisvar_stats.has_missing_values:
            msg = (
                'contains "missing" index values, which should not be present '
                "in a location-index-set."
            )
            log_lis("A402", msg)

        if "_FillValue" in lis_var.attributes:
            msg = (
                "has a '_FillValue' attribute, which should not be present "
                "on a location-index-set."
            )
            log_lis("A403", msg)

        if mesh_var and lis_dim and parent_dim:
            len_lis = self.file_scan.dimensions[lis_dim].length
            len_parent = self.file_scan.dimensions[parent_dim].length
            if len_lis >= len_parent:
                msg = (
                    f'has dimension "{lis_dim}", length {len_lis}, which is '
                    f"longer than the {location} dimension of the parent "
                    f'mesh "{mesh_name}" : '
                    f'"{parent_dim}", length {len_parent}.'
                )
                log_lis("A404", msg)

        if lisvar_stats.has_duplicate_values:
            msg = (
                "contains repeated index values, which should not be present "
                "in a location-index-set."
            )
            log_lis("A405", msg)

        min_ind = lisvar_stats.min_value
        if min_ind < index_offset:
            msg = (
                f"has some index value = {min_ind}, "
                f"which is less than the start-index value of {index_offset}."
            )
            log_lis("A406", msg)

        if parent_dim:
            dim = self.file_scan.dimensions.get(parent_dim)
            if dim:
                max_ind = lisvar_stats.max_value
                if (max_ind - index_offset) >= dim.length:
                    msg = (
                        f"contains a maximum index value of {max_ind}, "
                        "which is outside the range of the relevant "
                        f'"{parent_dim}" dimension, '
                        f"{index_offset}..{dim.length + index_offset - 1}."
                    )
                    log_lis("A406", msg)

    def dataset_identify_containers(self):
        """
        Find "mesh" , "mesh data", and "location index set" variables,

        Also include possibles due to mesh/lis references from data variables.

        Results set as self properties :
            self._meshdata_vars
            self._mesh_vars
            self._lis_vars
            self._mesh_referrers
            self._lis_referrers

        """
        # Location index sets are those with a cf_role of 'location_index_set'
        self._lis_vars = vars_w_props(
            self._all_vars, cf_role="location_index_set"
        )

        # Mesh data variables are those with either a 'mesh' or
        # 'location_index_set' attribute, but excluding the lis-vars.
        self._meshdata_vars = {
            varname: var
            for varname, var in self._all_vars.items()
            if (
                varname not in self._lis_vars
                and (
                    "mesh" in var.attributes
                    or "location_index_set" in var.attributes
                )
            )
        }
        # Mesh vars are those with cf_role="mesh_topology".
        self._mesh_vars = vars_w_props(self._all_vars, cf_role="mesh_topology")

        # Scan for any meshvars referred to by 'mesh' or 'location_index_set'
        # properties in mesh-data vars.
        # These are included among potential meshdata- and lis- variables
        # (so they are detected + checked even without the correct cf_role)
        self._mesh_referrers = {}
        self._lis_referrers = {}
        for referrer_name, referrer_var in list(self._meshdata_vars.items()):
            # Note: taking a copy as we may modify _meshdata_vars in the loop
            meshprop = referrer_var.attributes.get("mesh")
            meshvar_name = property_as_single_name(meshprop)
            if (
                meshvar_name is not None
                and meshvar_name in self._all_vars
                and meshvar_name not in self._mesh_vars
            ):
                # Add this reference to our list of all meshvars
                self._mesh_vars[meshvar_name] = self._all_vars[meshvar_name]
                # Record name of referring var.
                # N.B. potentially this can overwrite a previous referrer,
                # but "any one of several" will be OK for our purpose.
                self._mesh_referrers[meshvar_name] = referrer_name

            # Do something similar with lis references.
            meshprop = referrer_var.attributes.get("location_index_set")
            lisvar_name = property_as_single_name(meshprop)
            if (
                lisvar_name is not None
                and lisvar_name in self._all_vars
                and lisvar_name not in self._lis_vars
            ):
                # Add this reference to our list of all meshvars
                self._lis_vars[lisvar_name] = self._all_vars[lisvar_name]
                # Also remove it from the meshdata-vars if it was there
                # N.B. this could only happen if it has a wrong cf_role, but
                # that is just the kind of error we dealing with here.
                self._meshdata_vars.pop(lisvar_name, None)
                # Record name of referring var.
                self._lis_referrers[lisvar_name] = referrer_name

    def dataset_check_containers_and_map_dims(self):
        """
        Check all putative mesh + lis variables and collect dimension maps.
        Writes self._all_mesh_dims: {<mesh or lis name>: {location: dim-name}}

        Note: in checking the individual mesh variables, we also check all
        the coordinates and connectivities.

        This routine also sets self._allowed_cfrole_varnames

        """

        # Build a map of the dimensions of all the meshes,
        # all_meshes_dims: {meshname: {location: dimname}}
        self._all_mesh_dims = {}

        # This list of "UGRID variables" is used by 'dataset_global_checks' to
        # find any vars with a UGRID-style 'cf_role' that should not have one.
        # N.B. we don't include meshdata-variables, or coordinate variables,
        # which should *not* have a 'cf_role' anyway.
        # After this, all connectivities will be added by 'check_connectivity'.
        self._allowed_cfrole_varnames = list(self._mesh_vars.keys()) + list(
            self._lis_vars.keys()
        )

        # Find all connectivity variables and, initially, put them all on the
        # "orphan connectivities" list :  Those attached to meshes will be
        # removed when we check the meshes (next).
        self._orphan_connectivities = {
            var_name: var
            for var_name, var in self._all_vars.items()
            if (
                "cf_role" in var.attributes
                and (
                    str(var.attributes.get("cf_role"))
                    in _VALID_CONNECTIVITY_ROLES
                )
            )
        }

        # Check all mesh vars
        # Note: this call also fills in 'self._all_mesh_dims', and checks all
        # the attached coordinates and connectivites for each mesh.
        for meshvar in self._mesh_vars.values():
            self.check_mesh_var(meshvar)

        # Check all lis-vars
        # Note: this call also fills in 'self._all_mesh_dims'.
        for lis_var in self._lis_vars.values():
            self.check_lis_var(lis_var)

    def dataset_detect_shared_dims(self):
        """
        Check for any dimensions shared between meshes - an advisory warning.

        """
        # Convert all_meshes_dims: {meshname: {location: dimname}}
        # .. to dim_meshes: {dimname: [meshnames]}
        dim_meshes = {}
        for mesh, location_dims in self._all_mesh_dims.items():
            for location, dim in location_dims.items():
                # Fetch list
                meshnames = dim_meshes.get(dim, set())
                if dim:
                    # TODO: what if a dim is used by 2 different locations of
                    #  of the same mesh ?
                    meshnames.add(mesh)
                # Write list back
                dim_meshes[dim] = meshnames

        # Check for any dims which are used more than once.
        for dim, meshnames in dim_meshes.items():
            if len(meshnames) > 1:
                # TODO: what if a dim is used by 2 different locations of
                #  of the same mesh ?
                #  We would get a repeated meshname here...
                meshnames = sorted(meshnames)
                other_meshes, last_mesh = meshnames[:-1], meshnames[-1]
                if len(other_meshes) == 1:
                    other_mesh = other_meshes[0]
                    msg = (
                        f'Dimension "{dim}" is mapped by both '
                        f'mesh "{other_mesh}" and mesh "{last_mesh}".'
                    )
                else:
                    msg = f'Dimension "{dim}" is mapped by multiple meshes : '
                    msg += ", ".join(f'"{mesh}"' for mesh in other_meshes)
                    msg += f' and "{last_mesh}".'
                self.state("A104", None, None, msg)

    def dataset_detect_multiple_refs(self):
        """
        Check for any coords and conns referenced by multiple meshes.

        N.B. relevant errors are :
            * A201 coord should have 1-and-only-1 parent mesh
            * A301 connectivity should have 1-and-only-1 parent mesh

        """
        var_refs_meshes_attrs = {}
        all_ref_attrs = _VALID_MESHCOORD_ATTRS + _VALID_CONNECTIVITY_ROLES
        for some_meshname in sorted(self._mesh_vars):
            some_meshvar = self._mesh_vars[some_meshname]
            for some_refattr in all_ref_attrs:
                is_coord = some_refattr in _VALID_MESHCOORD_ATTRS
                attrval = some_meshvar.attributes.get(some_refattr, None)
                somevar_names = property_namelist(attrval)
                for somevar_name in somevar_names:
                    # NB only collect valid refs (to real variables)
                    if somevar_name in self._all_vars:
                        meshes = var_refs_meshes_attrs.get(somevar_name, set())
                        meshes.add((some_meshname, some_refattr))
                        var_refs_meshes_attrs[somevar_name] = meshes

        for some_varname, meshes_and_attrs in var_refs_meshes_attrs.items():
            some_var = self._all_vars[some_varname]  # NB have only 'real' refs
            if len(meshes_and_attrs) > 1:
                meshes_and_attrs = sorted(
                    meshes_and_attrs, key=lambda pair: pair[0]
                )
                refs_msg = ", ".join(
                    [
                        f"{some_mesh}:{attr_name}"
                        for some_mesh, attr_name in meshes_and_attrs
                    ]
                )
                msg = f"is referenced by multiple mesh variables : {refs_msg}."

                # Structurally, a var *could* be referenced as both a coord
                # *and* a connectivity.  But they have different required
                # numbers of dims, so we use that to decide what to call it.
                is_coord = len(some_var.dimensions) == 1
                if is_coord:
                    vartype = "Mesh coordinate"
                    code = "A201"
                else:
                    vartype = "Mesh connectivity"
                    code = "A301"
                self.state(code, vartype, some_varname, msg)

    def dataset_global_checks(self):
        """Do file-level checks not based on any particular variable type."""

        def log_dataset(errcode, msg):
            self.state(errcode, "", "", msg)

        # A901 "dataset contents should also be CF compliant" -- not checkable,
        # unless we integrate this code with cf-checker.

        # Check the global Conventions attribute for a UGRID version.
        conventions = self.file_scan.attributes.get("Conventions")
        if conventions is None:
            msg = ""
            log_dataset("A902", "dataset has no 'Conventions' attribute.")
        else:
            conventions = str(conventions)
            re_conventions = re.compile(r"UGRID-[0-9]+\.[0-9]+")
            if not re_conventions.search(conventions):
                # NOTE: just search.  Don't attempt to split, as usage of
                # comma/space/semicolon might be inconsistent, and we don't
                # need to care about that here.
                msg = (
                    f'dataset has Conventions="{conventions}", which does not '
                    "contain a UGRID convention statement of the form "
                    '"UGRID-<major>.<minor>".'
                )
                log_dataset("A903", msg)

        # Check for any unexpected 'cf_role' usages.
        # N.B. the logic here is that
        #   1) if it has a UGRID-type cf-role, then *either* it was already
        #       identified (and checked), *or* it generates a A904 warning
        #   2) if it has a CF cf-role, we don't comment
        #   3) if it has some other cf-role, this is unrecognised -> A905
        for var_name, var in self._all_vars.items():
            if (
                "cf_role" in var.attributes
                and var_name not in self._allowed_cfrole_varnames
            ):
                cf_role = str(var.attributes["cf_role"])
                if cf_role in _VALID_UGRID_CF_ROLES:
                    msg = (
                        f'has cf_role="{cf_role}", which is a UGRID defined '
                        "cf_role term, but the variable is not recognised as "
                        "a UGRID mesh, location-index-set or connectivity "
                        "variable."
                    )
                    self.state("A904", "netcdf", var_name, msg)
                elif cf_role not in _VALID_CF_CF_ROLES:
                    msg = (
                        f'has cf_role="{cf_role}", which is not a recognised '
                        "cf-role value defined by either CF or UGRID."
                    )
                    self.state("A905", "netcdf", var_name, msg)

    def check_dataset(self):
        """
        Run all conformance checks on the contained file scan.

        All results logged via `self.state`.

        """
        self.dataset_identify_containers()
        self.dataset_check_containers_and_map_dims()

        # Check any orphan connectivities.
        for var_name, var in self._orphan_connectivities.items():
            self.check_connectivity(var)
            # Always flag these as a possible problem.
            self.state("A301", "connectivity", var_name, "has no parent mesh.")

        # Check all the mesh-data vars
        for meshdata_var in self._meshdata_vars.values():
            self.check_meshdata_var(meshdata_var)

        # Do the checks which cut across different meshes
        self.dataset_detect_shared_dims()
        self.dataset_detect_multiple_refs()

        # Do the miscellaneous dataset-level checks
        self.dataset_global_checks()

    def checking_report(self) -> str:
        """Produce a text summary of the checking results."""
        report_lines = []

        def line(msg: str):
            report_lines.append(msg)

        log = self.logger
        logs = log.report_statement_logrecords()
        line("")
        line("UGRID conformance checks complete.")
        line("")
        if self.data_skipped:
            line(
                "WARNING: Some data checks skipped, "
                f"due to arrays larger than {self.max_mb_checks} Mb."
            )
            line("")
        if log.N_FAILURES + log.N_WARNINGS == 0:
            line("No problems found.")
        else:
            if logs:
                line("List of checker messages :")
                for log_record in logs:
                    line("  " + log_record.msg)
                line("")
            line(
                f"Total of {log.N_WARNINGS + log.N_FAILURES} "
                "problems logged :"
            )
            line(f"  {log.N_FAILURES} Rxxx requirement failures")
            line(f"  {log.N_WARNINGS} Axxx advisory recommendation warnings")
        line("")
        line("Done.")

        return "\n".join(report_lines)

    def _structure_reporter(self):
        reporter = StructureReporter(
            all_file_dims=self.file_scan.dimensions,
            all_file_vars=self.file_scan.variables,
            all_file_attrs=self.file_scan.attributes,
            mesh_location_dims=self._all_mesh_dims,
            mesh_vars=self._mesh_vars,
            lis_vars=self._lis_vars,
            meshdata_vars=self._meshdata_vars,
            orphan_connectivities=self._orphan_connectivities,
        )
        return reporter

    def structure_report(self, include_nonmesh: bool = False) -> str:
        """
        Produce a text summary of the dataset UGRID structure.

        Parameters
        ----------
        include_nonmesh : bool, default False
            If set, also output a list of file dimensions and variables *not*
            relating to the UGRID meshes contained.

        """
        # Implemented in separate 'reporter' class to organise the code better.
        # Create one, passing it all our key analysis info.
        reporter = self._structure_reporter()
        return reporter.report(include_nonmesh=include_nonmesh)

    def structure_object(self):
        reporter = self._structure_reporter()
        return reporter.structure_info()


class StructureReporter:
    """
    A class which can produce a structure report on a dataset.

    This is based on data provided by a Checker, to identify the variables
    by role and build a dimensions map.
    Provided as a separate class purely to enable clearer coding.

    """

    indent = "    "

    def __init__(
        self,
        all_file_dims: Mapping[Text, NcDimSummary],
        all_file_vars: Mapping[Text, NcVariableSummary],
        all_file_attrs: Mapping[Text, np.ndarray],
        mesh_location_dims: Dict[str, Dict[str, Union[None, str]]],
        mesh_vars: Dict[str, NcVariableSummary],
        lis_vars: Dict[str, NcVariableSummary],
        meshdata_vars: Dict[str, NcVariableSummary],
        orphan_connectivities: Dict[str, NcVariableSummary],
    ):
        self.all_file_dims = all_file_dims
        self.all_file_vars = all_file_vars
        self.all_file_attrs = all_file_attrs
        self.mesh_location_dims = mesh_location_dims
        self.mesh_vars = mesh_vars
        self.lis_vars = lis_vars
        self.orphan_connectivities = orphan_connectivities
        self.meshdata_vars = meshdata_vars
        # Internals
        self.result_lines = []
        self._struct: UgridDataset = UgridDataset()

    def _line(self, msg: Text, n_indent: int = 0):
        self.result_lines.append(self.indent * n_indent + msg)

    @staticmethod
    def _varlist_str(var: NcVariableSummary, attr_name: str) -> str:
        names_attr = var.attributes.get(attr_name)
        if not names_attr:
            result = "? <none>"
        else:
            names = str(names_attr).split(" ")
            result = ", ".join(f'"{name}"' for name in names)
        return result

    def structure_info(self):
        """Construct + return a structure object."""
        self._struct = UgridDataset(global_attributes=self.all_file_attrs)
        self._struct.dims.all = self.all_file_dims
        self._struct.vars.all = self.all_file_vars
        self._mesh_report()
        self._nonmesh_calculate()
        return self._struct

    def report(self, include_nonmesh: bool = False) -> str:
        """
        Produce a text summary of dataset structure.

        Parameters
        ----------
        include_nonmesh : bool, default False
            If set, also output a list of file dimensions and variables *not*
            relating to the UGRID meshes contained.

        """
        self.result_lines = []
        self._mesh_report()
        if include_nonmesh:
            self._nonmesh_report()
        return "\n".join(self.result_lines)

    def _valid_var(self, var_name: str):
        # Return var or None
        return self.all_file_vars.get(str(var_name))

    def _valid_vars_map(self, var_names: List[str]):
        # Return dict of {name:var}, only valid vars
        varsdict = {}
        for name in var_names:
            var = self.all_file_vars.get(str(name))
            if name is not None:
                varsdict[name] = var
        return varsdict

    def _mesh_report(self):
        if not self.mesh_vars:
            self._line("Meshes : <none>")
        else:
            self._line("Meshes")
            for mesh_name, mesh_var in self.mesh_vars.items():
                struct_mesh = UgridMesh(name=mesh_name, var=mesh_var)
                self._line(f'"{mesh_name}"', 1)
                dims = self.mesh_location_dims[mesh_name]
                # Nodes is a bit 'special'
                dim = dims["node"]
                if not dim:
                    self._line("<? no node coordinates or dimension ?>", 2)
                    struct_mesh = None
                else:
                    self._line(f'node("{dim}")', 2)
                    coords = self._varlist_str(mesh_var, "node_coordinates")
                    self._line(f"coordinates : {coords}", 3)
                    coords = mesh_var.attributes.get("node_coordinates", "")
                    coords = str(coords).split()
                    struct_mesh.coords.node = self._valid_vars_map(coords)
                # Other dims all reported in the same way
                for location in ("edge", "face", "boundary"):
                    dim = dims[location]
                    if dim:
                        self._line(f'{location}("{dim}")', 2)
                        connattr_name = f"{location}_node_connectivity"
                        conn_str = self._varlist_str(mesh_var, connattr_name)
                        self._line(f"{connattr_name} : {conn_str}", 3)
                        coords_attrname = f"{location}_coordinates"
                        if coords_attrname in mesh_var.attributes:
                            coords = self._varlist_str(
                                mesh_var, coords_attrname
                            )
                            self._line(f"coordinates : {coords}", 3)

                        if struct_mesh is not None:
                            conn_name = mesh_var.attributes.get(connattr_name)
                            conn = self._valid_var(conn_name)
                            struct_mesh.conns[connattr_name] = conn
                            coords_attr = mesh_var.attributes.get(
                                coords_attrname
                            )
                            coord_varnames = property_namelist(coords_attr)
                            coords = self._valid_vars_map(coord_varnames)
                            setattr(struct_mesh.coords, location, coords)
                # List *optional* connectivities of the mesh
                # N.B. the "<x>_node..." are required connectivities, which
                # are displayed above, under their locations.
                # Now display any others (optional) in a separate section.
                optional_conns = []
                for connattr_name in _VALID_CONNECTIVITY_ROLES:
                    if connattr_name.split("_")[1] != "node":
                        attr_value = property_as_single_name(
                            mesh_var.attributes.get(connattr_name)
                        )
                        if attr_value:
                            if attr_value in self.all_file_vars:
                                attr_text = f'"{attr_value}"'
                                if struct_mesh is not None:
                                    struct_mesh.conns[
                                        connattr_name
                                    ] = self._valid_var(attr_value)
                            else:
                                attr_text = f'<?nonexistent?> "{attr_value}"'
                            optional_conns.append((connattr_name, attr_text))

                if optional_conns:
                    self._line("optional connectivities", 2)
                    for connattr_name, attr_text in optional_conns:
                        self._line(f"{connattr_name} : {attr_text}", 3)

            if struct_mesh is not None:
                self._struct.meshes[mesh_name] = struct_mesh

        if self.lis_vars:
            self._line("")
            self._line("Location Index Sets")
            for lis_name, lis_var in self.lis_vars.items():
                mesh = lis_var.attributes.get("mesh")
                if mesh is not None:
                    mesh = str(mesh)
                location = lis_var.attributes.get("location")
                if location is not None:
                    location = str(location)
                dims = self.mesh_location_dims.get(lis_name, {})
                dim_name = dims.get(str(location))
                valid = (
                    mesh in self._struct.meshes
                    and location is not None
                    and dims is not None
                )
                if valid:
                    self._struct.lises[lis_name] = UgridLis(
                        name=lis_name,
                        mesh=self._struct.meshes[mesh_name],
                        location=location,
                        datavars={},
                        var=lis_var,
                    )
                if location is None:
                    location = "? <none>"
                if mesh is None:
                    mesh = "? <none>"
                if dim_name is None:
                    dim_name = "? <none>"
                self._line(f'"{lis_name}" ({dim_name})', 1)
                self._line(f'mesh : "{mesh}"', 2)
                self._line(f"location : {location}", 2)

        if self.orphan_connectivities:
            self._line("")
            self._line("?? Connectivities with no mesh ??")
            for conn_name, conn_var in self.orphan_connectivities.items():
                dims = ", ".join(f'"{dim}"' for dim in conn_var.dimensions)
                self._line(f'"{conn_name}" ({dims})', 1)
                cf_role = self._varlist_str(conn_var, "cf_role")
                self._line(f"cf_role = {cf_role}", 2)

        if self.meshdata_vars:
            self._line("")
            self._line("Mesh Data Variables")
            for var_name, var in self.meshdata_vars.items():
                self._line(f'"{var_name}"', 1)
                attrs = {
                    attr_name: var.attributes.get(attr_name)
                    for attr_name in ("mesh", "location", "location_index_set")
                }
                # 'treat as' mirrors logic in 'Checker.check_meshdata_var'
                treat_as_lis = attrs["location_index_set"] and (
                    not attrs["mesh"] or not attrs["location"]
                )
                if treat_as_lis:
                    order_and_expected = [
                        ("location_index_set", True),
                        ("mesh", False),
                        ("location", False),
                    ]
                else:
                    order_and_expected = [
                        ("mesh", True),
                        ("location", True),
                        ("location_index_set", False),
                    ]
                valid = False
                for connattr_name, expected in order_and_expected:
                    attr = attrs[connattr_name]
                    value = None
                    if attr:
                        value = self._varlist_str(var, connattr_name)
                        if not expected:
                            value = f"<?unexpected?> {value}"
                            valid = False
                    elif expected:
                        value = "<?missing?>"
                        valid = False
                    if value:
                        self._line(f"{connattr_name} : {value}", 2)

                valid = True
                mesh = attrs["mesh"]
                location = attrs["location"]
                lis = attrs["location_index_set"]
                if mesh is not None:
                    mesh = self._struct.meshes.get(str(mesh))
                    if mesh is None:
                        valid = False
                if lis is not None:
                    lis = self._struct.lises.get(str(mesh))
                    if lis is None:
                        valid = False
                if lis is not None:
                    valid = mesh is None and location is None
                else:
                    valid = mesh is not None and location is not None
                    if valid:
                        location = str(location)
                        valid = location in _VALID_UGRID_LOCATIONS
                if valid:
                    self._struct.datavars[var_name] = UgridDatavar(
                        name=var_name,
                        lis=lis,
                        mesh=mesh,
                        location=location,
                        var=var,
                    )

    def _nonmesh_calculate(self, output_struct: UgridDataset = None):
        # A non-mesh var is one that is not referenced by any UGRID mesh
        # components.
        all_mesh_varnames = (
            set(self.mesh_vars.keys())
            | set(self.lis_vars.keys())
            | set(self.meshdata_vars.keys())
            | set(self.orphan_connectivities.keys())
        )
        nonmesh_vars = set(self.all_file_vars.keys()) - all_mesh_varnames

        # A mesh dimension is one that is a location dim of any
        # mesh, or any connectivity (e.g. includes dims used for
        # nodes of a face).
        nonmesh_dims = set(self.all_file_dims.keys())

        # Exclude from 'nonmesh' : all dims and vars of each mesh.
        for meshvar in self.mesh_vars.values():
            # Exclude all mesh location dims.
            mesh_dims = self.mesh_location_dims[meshvar.name]
            nonmesh_dims -= set(mesh_dims.values())

            # Exclude all location coordinates, and their bounds vars.
            for location in _VALID_UGRID_LOCATIONS:
                attrname = f"{location}_coordinates"
                attr = meshvar.attributes.get(attrname)
                location_coord_names = property_namelist(attr)
                nonmesh_vars -= set(location_coord_names)
                for coord_name in location_coord_names:
                    coord_var = self.all_file_vars.get(coord_name)
                    bounds_attr = coord_var.attributes.get("bounds")
                    bounds_varname = property_as_single_name(bounds_attr)
                    if bounds_varname:
                        nonmesh_vars.discard(bounds_varname)

            # Exclude all connectivities, and all their dims.
            for attrname in _VALID_CONNECTIVITY_ROLES:
                conn_attr = meshvar.attributes.get(attrname)
                conn_name = property_as_single_name(conn_attr)
                if conn_name:
                    nonmesh_vars.discard(conn_name)
                    conn_var = self.all_file_vars.get(conn_name)
                    if conn_var:
                        nonmesh_dims -= set(conn_var.dimensions)

        # Also exclude all dimensions of 'orphan' connectivities.
        for conn_var in self.orphan_connectivities.values():
            nonmesh_dims -= set(conn_var.dimensions)

        self._nonmesh_dims = nonmesh_dims
        self._nonmesh_vars = nonmesh_vars

    def _nonmesh_report(self):
        self._nonmesh_calculate()
        # Add report section, if any nonmesh found.
        if self._nonmesh_dims or self._nonmesh_vars:
            self._line("")
            self._line("Non-mesh variables and/or dimensions")
            if self._nonmesh_dims:
                self._line("dimensions:", 1)
                for dim in sorted(self._nonmesh_dims):
                    self._line(f'"{dim}"', 2)
            if self._nonmesh_vars:
                self._line("variables:", 1)
                for var in sorted(self._nonmesh_vars):
                    self._line(f'"{var}"', 2)


def check_dataset(
    file: Union[NcFileSummary, AnyStr, Path],
    print_summary: bool = True,
    omit_advisories: bool = False,
    ignore_codes: Union[List[str], None] = None,
    max_data_mb: float = 200.0,
) -> Checker:
    """
    Run UGRID conformance checks on a file.

    Optionally print a result summary.
    Optionally ignore errors below a logging level.
    Returns a checker object with a file analysis and checking log records.

    Parameters
    ----------
    file : string, Path or :class:`NcFileSummary`
        path to, or representation of a netcdf input file
    print_summary : bool, default=True
        print a results summary at the end
    omit_advisories : bool, default False
        If set, log only 'requirements' Rxxx statements, and ignore the
        advisory 'Axxx' ones.
    ignore_codes : list(str) or None, default None
        A list of error codes to ignore.
    max_data_mb : float, default 200.0
        A rough size threshold (in Mb), beyond which we will skip data checks.
        Set to 0 for no checking, or -1 for no size limit.

    Returns
    -------
    checker : Checker
        A checker for the file.

    """
    if isinstance(file, str):
        file_path = Path(file)
    elif isinstance(file, Path):
        file_path = file
    if isinstance(file, NcFileSummary):
        file_scan = file
    else:
        file_scan = scan_dataset(file_path)

    checker = Checker(
        file_scan,
        ignore_codes=ignore_codes,
        ignore_warnings=omit_advisories,
        max_mb_checks=max_data_mb,
    )

    if print_summary:
        # Print the results : this is the default action
        print(checker.checking_report())

    return checker
