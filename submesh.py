"""Sub-mesh generation utilities for sparse sensor placement.

Provides Delaunay triangulation-based sub-sampling of unstructured meshes
for constructing sparse observation graphs used in the PiTDN-ROM framework.
"""

import numpy as np


def get_edges(tri):
    """Extract unique undirected edges from a Delaunay triangulation.

    Args:
        tri: A scipy.spatial.Delaunay triangulation object.

    Returns:
        Set of unique edges as sorted tuples.
    """
    edges = set()
    for simplex in tri.simplices:
        v1, v2, v3 = simplex
        edges.add(tuple(sorted([v1, v2])))
        edges.add(tuple(sorted([v2, v3])))
        edges.add(tuple(sorted([v3, v1])))
    return edges
