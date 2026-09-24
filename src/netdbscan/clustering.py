"""Deterministic DBSCAN on a precomputed road-distance relation.

DBSCAN only asks one question about a pair of observations: is their distance
at most ``cluster_eps``? The input can therefore be either of two forms:

* the sparse ``(n, n)`` matrix from ``road.neighbor_graph``: stored entries
  are the pairs within ``cluster_eps`` (a stored zero is a neighbour), and a
  missing entry is a pair that is farther apart or disconnected;
* a dense ``(n, n)`` matrix in which ``+inf`` marks disconnected pairs (for
  tests and small studies).

A dense matrix is converted to the sparse form before DBSCAN: every entry
``<= cluster_eps`` is kept and every other entry, ``+inf`` included, is
dropped. So no finite stand-in for infinity is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, issparse
from sklearn.cluster import DBSCAN
from sklearn.neighbors import sort_graph_by_row_values

NoisePolicy = Literal["exclude", "singleton"]


@dataclass(frozen=True)
class ClusteringResult:
    """DBSCAN output in canonical point order (see ``canonical_point_order``).

    ``is_core`` marks DBSCAN core points. A clustered point that is not a core
    point is a border point: it is within ``cluster_eps`` of a core point but
    has too few neighbours itself. When a border point is within reach of two
    clusters, DBSCAN gives it to the cluster that is expanded first, which
    here is decided by the canonical order.
    """

    point_ids: list[object]
    point_keys: list[str]
    raw_labels: np.ndarray
    cluster_ids: np.ndarray
    is_noise: np.ndarray
    is_core: np.ndarray

    @property
    def n_noise(self) -> int:
        return int(self.is_noise.sum())

    @property
    def n_core(self) -> int:
        return int(self.is_core.sum())

    @property
    def n_clusters(self) -> int:
        return len({x for x in self.cluster_ids.tolist() if x is not None})

    @property
    def n_clustered(self) -> int:
        return int(sum(x is not None for x in self.cluster_ids.tolist()))


def canonical_point_order(point_ids: Sequence[object]) -> np.ndarray:
    """Row order sorted by ``str(point_id)`` (Python string order: ``"10" < "2"``)."""
    values = pd.Series(list(point_ids), dtype=object)
    if values.isna().any():
        raise ValueError("point_id values must not be null")
    keys = values.map(str)
    if keys.duplicated().any():
        raise ValueError("point_id values must be unique after string conversion")
    return np.argsort(keys.to_numpy(dtype=object), kind="stable")


def _check_parameters(cluster_eps: float, min_samples: int, noise_policy: str) -> float:
    if noise_policy not in {"exclude", "singleton"}:
        raise ValueError("noise_policy must be 'exclude' or 'singleton'")
    cluster_eps = float(cluster_eps)
    if not (np.isfinite(cluster_eps) and cluster_eps > 0):
        raise ValueError("cluster_eps must be a positive finite number")
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
        raise ValueError("min_samples must be an integer >= 1")
    return cluster_eps


def _radius_graph_from_dense(distances: np.ndarray, cluster_eps: float) -> csr_matrix:
    D = np.asarray(distances, dtype=float)
    if np.isnan(D).any() or np.isneginf(D).any():
        raise ValueError("distance matrix must not contain NaN or -inf")
    if (D < 0).any():
        raise ValueError("distance matrix must not contain negative distances")
    if not np.allclose(D, D.T):
        raise ValueError("distance matrix must be symmetric")
    # Differences in the last bits are resolved as in ``road.neighbor_graph``:
    # each pair gets the smaller of its two values.
    D = np.minimum(D, D.T)
    within = D <= cluster_eps
    np.fill_diagonal(within, False)
    rows, cols = np.nonzero(within)
    return coo_matrix((D[rows, cols], (rows, cols)), shape=D.shape).tocsr()


def _check_radius_graph(graph, n: int) -> csr_matrix:
    graph = csr_matrix(graph, dtype=float)
    if graph.shape != (n, n):
        raise ValueError(f"neighbour matrix must have shape {(n, n)}, got {graph.shape}")
    if not np.isfinite(graph.data).all() or (graph.data < 0).any():
        raise ValueError("stored neighbour distances must be finite and >= 0")
    # Compare stored entries directly: a subtraction such as ``graph - graph.T``
    # would not notice a stored zero that has no mirror entry.
    coo = graph.tocoo()
    forward = np.lexsort((coo.col, coo.row))
    backward = np.lexsort((coo.row, coo.col))
    same_pairs = np.array_equal(coo.row[forward], coo.col[backward]) and np.array_equal(
        coo.col[forward], coo.row[backward]
    )
    if not same_pairs or not np.array_equal(coo.data[forward], coo.data[backward]):
        raise ValueError("neighbour matrix must be symmetric, stored entries included")
    return graph


def _canonical_graph(graph: csr_matrix, order: np.ndarray) -> csr_matrix:
    """Reorder rows and columns, and store every diagonal entry as an explicit zero.

    The work is done on COO triplets because SciPy's sparse addition drops
    stored zeros, and for DBSCAN a stored zero is a neighbour. DBSCAN stores
    the diagonal itself as well; doing it here also keeps scikit-learn's
    ``sort_graph_by_row_values`` working when no pair is within ``cluster_eps``
    (it raises ``IndexError`` on a graph with no stored entries).
    """
    n = len(order)
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)
    coo = graph.tocoo()
    off_diagonal = coo.row != coo.col
    rows = np.concatenate([position[coo.row[off_diagonal]], np.arange(n)])
    cols = np.concatenate([position[coo.col[off_diagonal]], np.arange(n)])
    data = np.concatenate([coo.data[off_diagonal], np.zeros(n)])
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def cluster_precomputed(
    point_ids: Sequence[object],
    distances,
    *,
    cluster_eps: float,
    min_samples: int,
    noise_policy: NoisePolicy = "exclude",
    position: Sequence[int] | np.ndarray | None = None,
) -> ClusteringResult:
    """Run deterministic DBSCAN on road distances between the points.

    Without ``position``, ``distances`` has one row per point: ``(n, n)``.

    With ``position``, several points can share one row. ``position[k]`` is
    the row of point ``k`` (rows ``0 .. P - 1``, each used at least once), and
    ``distances`` is ``(P, P)``. Points that share a position have distance 0
    to each other and the same distance to every other point, so DBSCAN runs
    on the positions with ``sample_weight`` equal to the number of points at
    each position: a position is a core point when the weights within
    ``cluster_eps`` sum to at least ``min_samples``. This gives the same labels
    and core flags as DBSCAN on all points, without the k(k - 1)/2
    zero-distance pairs of k points at one position.

    Rows are first put in canonical order (points by ``str(point_id)``,
    positions by their smallest member key), so the result does not depend on
    the input row order. Public cluster IDs are numbered in the order of each
    cluster's smallest member key: ``C000001``, ``C000002``, ...
    """
    cluster_eps = _check_parameters(cluster_eps, min_samples, noise_policy)
    ids = list(point_ids)
    n = len(ids)
    if position is None:
        position = np.arange(n, dtype=np.int64)
    else:
        position = np.asarray(position)
        if position.shape != (n,) or (n and not np.issubdtype(position.dtype, np.integer)):
            raise ValueError("position must hold one integer per point")
        position = position.astype(np.int64)
    message = "position must number the positions 0 .. P - 1, each used by at least one point"
    if n and position.min() < 0:
        raise ValueError(message)
    n_positions = int(position.max()) + 1 if n else 0
    weight = np.bincount(position, minlength=n_positions) if n else np.empty(0, dtype=np.int64)
    if (weight == 0).any():
        raise ValueError(message)
    if issparse(distances):
        graph = _check_radius_graph(distances, n_positions)
    else:
        dense = np.asarray(distances, dtype=float)
        if dense.shape != (n_positions, n_positions):
            raise ValueError(f"distance matrix must have shape {(n_positions, n_positions)}, got {dense.shape}")
        graph = _radius_graph_from_dense(dense, cluster_eps)

    order = canonical_point_order(ids)
    ids = [ids[i] for i in order]
    keys = [str(v) for v in ids]
    empty = np.empty(0, dtype=object)
    if n == 0:
        return ClusteringResult([], [], np.empty(0, dtype=int), empty, np.empty(0, bool), np.empty(0, bool))

    # Canonical position order: by the canonical rank of each position's first point.
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    first_rank = np.full(n_positions, n, dtype=np.int64)
    np.minimum.at(first_rank, position, rank)
    position_order = np.argsort(first_rank, kind="stable")
    graph = _canonical_graph(graph, position_order)
    graph = sort_graph_by_row_values(graph, copy=False, warn_when_not_sorted=False)
    weight = weight[position_order]
    model = DBSCAN(eps=cluster_eps, min_samples=int(min_samples), metric="precomputed").fit(
        graph, sample_weight=None if (weight == 1).all() else weight
    )

    # Back from canonical positions to points in canonical order.
    canonical_position = np.empty(n_positions, dtype=np.int64)
    canonical_position[position_order] = np.arange(n_positions)
    of_point = canonical_position[position[order]]
    raw = model.labels_[of_point]
    noise = raw == -1
    core_position = np.zeros(n_positions, dtype=bool)
    core_position[model.core_sample_indices_] = True
    core = core_position[of_point]

    groups: list[tuple[str, np.ndarray]] = []
    for label in sorted(set(raw.tolist()) - {-1}):
        members = np.flatnonzero(raw == label)
        groups.append((min(keys[i] for i in members), members))
    if noise_policy == "singleton":
        for i in np.flatnonzero(noise):
            groups.append((keys[i], np.asarray([i], dtype=int)))
    groups.sort(key=lambda item: item[0])

    public = np.empty(n, dtype=object)
    public[:] = None
    for number, (_, members) in enumerate(groups, start=1):
        public[members] = f"C{number:06d}"
    return ClusteringResult(ids, keys, raw, public, noise, core)
