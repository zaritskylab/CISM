import warnings

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree


def build_spatial_edge_pairs(
    coordinates: np.ndarray,
    max_distance: float,
    *,
    context: str = "FOV",
) -> list[tuple[int, int]]:
    """Build undirected spatial edges with a safe radius fallback."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"{context}: coordinates must have shape (N, 2); received "
            f"{coordinates.shape}."
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{context}: coordinates contain non-finite values.")
    if max_distance < 0:
        raise ValueError(f"{context}: max_distance must be non-negative.")

    num_nodes = coordinates.shape[0]
    if num_nodes < 2:
        return []

    unique_count = np.unique(coordinates, axis=0).shape[0]
    centered_rank = np.linalg.matrix_rank(coordinates - coordinates[0])
    use_radius_fallback = (
        num_nodes < 3
        or unique_count != num_nodes
        or centered_rank < 2
    )

    if not use_radius_fallback:
        try:
            triangulation = Delaunay(coordinates)
            indptr, neighbours = triangulation.vertex_neighbor_vertices
            pairs = {
                tuple(sorted((source, int(target))))
                for source in range(num_nodes)
                for target in neighbours[indptr[source]:indptr[source + 1]]
                if source != target
                and np.linalg.norm(
                    coordinates[source] - coordinates[target]
                ) <= max_distance
            }
            return sorted(pairs)
        except QhullError as error:
            warnings.warn(
                f"{context}: Delaunay triangulation failed; using "
                f"radius-neighbor edges ({error.args[0]}).",
                RuntimeWarning,
                stacklevel=2,
            )

    return sorted(cKDTree(coordinates).query_pairs(max_distance))
