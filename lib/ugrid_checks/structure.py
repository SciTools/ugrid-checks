import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from ugrid_checks.nc_dataset_scan import NcDimSummary, NcVariableSummary

# Basic classes with convenience names
Varname = str
# Location = Literal["node", "edge", "face"]
# Role = Literal[
#     "edge_node_connectivity",
#     "face_node_connectivity",
#     "face_edge_connectivity",
#     "edge_face_connectivity",
#     "face_face_connectivity",
#     "boundary_node_connectivity",
# ]
# Dropped for now : "Literal" types not available in Python 3.7
Location = str
Role = str
Var = NcVariableSummary
Dim = NcDimSummary
VarsMap = Dict[Varname, Var]
DimsMap = Dict[Varname, Dim]
Attrs = Dict[Varname, np.ndarray]


# Assign this to make a dataclass field default to an empty dict.
# Required for dataclass fields, as mutable init values are not permitted.
def emptydict():
    return dataclasses.field(default_factory=dict)


@dataclass()
class UgridDatavar:
    """
    Represents a mesh-data-variable.
    """

    name: str
    var: Var  # original low-level data
    lis: Optional["UgridLis"] = None
    mesh: Optional["UgridMesh"] = None
    location: Optional[Location] = None


@dataclass
class _CoordsLocationMap:
    node: VarsMap = emptydict()
    edge: VarsMap = emptydict()
    face: VarsMap = emptydict()


@dataclass()
class UgridMesh:
    """
    Represents a mesh.
    """

    name: str
    var: Var  # original low-level data
    coords: _CoordsLocationMap = _CoordsLocationMap()
    conns: Dict[Role, Var] = emptydict()
    # Additional summary info
    location_dims: Dict[Location, Dim] = emptydict()
    all_dims: DimsMap = emptydict()
    lisets: Dict[Varname, "UgridLis"] = emptydict()
    datavars: Dict[Varname, UgridDatavar] = emptydict()


@dataclass()
class UgridLis:
    """
    Represents a location-index-set.
    """

    name: str
    var: Var
    mesh: UgridMesh
    location: Location
    # Additional summary info
    datavars: Dict[Varname, UgridDatavar] = emptydict()


@dataclass
class _Dims_MeshNonmeshAll:
    all: DimsMap = emptydict()
    mesh: DimsMap = emptydict()
    nonmesh: DimsMap = emptydict()


@dataclass
class _Vars_MeshNonmeshAll:
    all: VarsMap = emptydict()
    mesh: VarsMap = emptydict()
    nonmesh: VarsMap = emptydict()


@dataclass()
class UgridDataset:
    """
    Represents the mesh structure of an entire dataset.
    """

    dims: _Dims_MeshNonmeshAll = _Dims_MeshNonmeshAll()
    vars: _Vars_MeshNonmeshAll = _Vars_MeshNonmeshAll()
    meshes: Dict[Varname, UgridMesh] = emptydict()
    lises: Dict[Varname, UgridLis] = emptydict()
    datavars: Dict[Varname, UgridDatavar] = emptydict()
    global_attributes: Dict[Varname, Attrs] = emptydict()
