"""Road-network snapping and bounded shortest-path neighbourhood search.

The public pipeline clusters observations by shortest-path distance along a
supplied line network.  Points are snapped to continuous positions on road
arcs with PySAL ``spaghetti``; DBSCAN then receives only the sparse pairs whose
network distance is at most ``eps``.

The point-to-network snap distance is quality-assurance metadata and is not
added to the DBSCAN metric.  Disconnected road components therefore have
infinite conceptual distance and never become neighbours.

The implementation avoids ``spaghetti.Network.allneighbordistances`` because
that routine first constructs a dense road-vertex distance matrix.  Instead,
bounded SciPy Dijkstra searches discover only pairs within ``eps``.  Repeated
observations at exactly the same snapped network position are represented once
for the neighbour search and later passed to DBSCAN with their multiplicity as
``sample_weight``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

# Upper limit for the number of neighbour pairs (unordered pairs of snapped
# positions). Measured peak memory: about 90 bytes per pair while the pairs are
# found and sorted, and about 160 bytes per pair during DBSCAN (the matrix plus
# scikit-learn's neighbourhood lists). The default therefore allows about 1.6 GB.
DEFAULT_MAX_NEIGHBOR_PAIRS = 10_000_000

# Observation indices in the pair lists; 32 bits suffice below 2**31 observations.
_INDEX = np.int32

# Memory for the dense result of one batch of bounded searches: each search
# returns one float64 per node of the search graph.
_SEARCH_BATCH_BYTES = 256 * 2**20

# Reached (search, road vertex) entries handled at one time, and candidate
# pairs formed at one time, when search results are joined with the points on
# the arcs at each reached vertex. Both bound temporary memory (about 100 bytes
# per entry).
_REACHED_CHUNK = 2_000_000
_CANDIDATE_CHUNK = 2_000_000

# Same-arc pairs produced at one time. Much smaller than the pair limit, so a
# dense arc never materialises a large pair array at once.
_SAME_ARC_BATCH_PAIRS = 250_000


def _spaghetti_module():
    try:
        import spaghetti
    except ImportError as exc:  # pragma: no cover - dependency installed in normal use
        raise ImportError("spaghetti is required for road-network clustering") from exc
    return spaghetti


def _can_skip_component_labelling(spaghetti_module) -> bool:
    """Whether the audited fast network build may be used.

    ``spaghetti`` labels connected components while it builds a network, and
    in 1.7.6 this is by far the slowest part of the build (measured: 9.7 s
    instead of 0.09 s for a 3,600-vertex grid). Snapping reads the component
    table only to fill ``pointpattern.component_to_obs``, which this package
    never reads (``spaghetti/network.py``, lines 1271-1280 in 1.7.6).
    The fast path is guarded to the exact audited ``spaghetti`` version. Any
    other version uses the ordinary public path until it has been checked.
    """
    return str(getattr(spaghetti_module, "__version__", "")) == "1.7.6"


# ---------------------------------------------------------------------------
# The road graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoadGraph:
    """The road network of the clustering stage.

    ``arc_u < arc_v`` for every arc; arcs from a vertex to itself are left
    out because they have no length. ``component[k]`` is the connected
    component of vertex ``k``.
    """

    network: object  # the spaghetti.Network, kept for snapping
    vertex_xy: np.ndarray
    arc_u: np.ndarray
    arc_v: np.ndarray
    arc_length: np.ndarray
    adjacency: csr_matrix
    component: np.ndarray

    @property
    def n_vertices(self) -> int:
        return len(self.vertex_xy)

    @cached_property
    def arc_index(self) -> shapely.STRtree:
        """Spatial index of the arcs, built on first use (for ``components_at``)."""
        lines = shapely.linestrings(np.stack([self.vertex_xy[self.arc_u], self.vertex_xy[self.arc_v]], axis=1))
        return shapely.STRtree(lines)


def _min_weight_csr(rows, cols, weights, shape) -> csr_matrix:
    """Sparse matrix that keeps the smallest weight of each (row, col) pair.

    Duplicate entries are removed before conversion, because a plain COO to
    CSR conversion adds duplicate weights together. Zero weights stay stored:
    for SciPy's shortest-path functions a stored zero is an edge of length 0.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    order = np.lexsort((weights, cols, rows))
    rows, cols, weights = rows[order], cols[order], weights[order]
    first = np.ones(len(rows), dtype=bool)
    first[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    return coo_matrix((weights[first], (rows[first], cols[first])), shape=shape).tocsr()


def build_road_graph(roads: gpd.GeoDataFrame) -> RoadGraph:
    """Build the road graph once with ``spaghetti``.

    ``spaghetti`` joins two road lines only where they share a vertex after
    rounding each coordinate to 11 significant digits (its ``vertex_sig``
    default). No other connection is inferred.
    """
    spaghetti = _spaghetti_module()
    skip_components = _can_skip_component_labelling(spaghetti)
    network = spaghetti.Network(
        in_data=roads,
        unique_arcs=True,
        extractgraph=False,
        w_components=not skip_components,
    )
    if skip_components:
        network.network_component2arc = {}

    n_vertices = len(network.vertex_coords)
    if sorted(network.vertex_coords) != list(range(n_vertices)):
        raise RuntimeError("spaghetti vertex IDs are not 0, 1, ..., n - 1")
    vertex_xy = np.array([network.vertex_coords[k][:2] for k in range(n_vertices)], dtype=float)

    arcs = np.array(
        sorted({tuple(sorted((int(a), int(b)))) for a, b in network.arcs}), dtype=np.int64
    ).reshape(-1, 2)
    proper = arcs[:, 0] != arcs[:, 1]
    arc_u, arc_v = arcs[proper, 0], arcs[proper, 1]
    if len(arc_u) == 0:
        raise ValueError("spaghetti produced no positive-length road arcs")
    arc_length = np.hypot(*(vertex_xy[arc_v] - vertex_xy[arc_u]).T)
    if not (np.isfinite(arc_length).all() and (arc_length > 0).all()):
        raise RuntimeError("two distinct spaghetti vertices have the same coordinates")

    adjacency = _min_weight_csr(
        np.concatenate([arc_u, arc_v]),
        np.concatenate([arc_v, arc_u]),
        np.concatenate([arc_length, arc_length]),
        shape=(n_vertices, n_vertices),
    )
    _, component = connected_components(adjacency, directed=False)
    return RoadGraph(
        network=network,
        vertex_xy=vertex_xy,
        arc_u=arc_u,
        arc_v=arc_v,
        arc_length=arc_length,
        adjacency=adjacency,
        component=component,
    )


# ---------------------------------------------------------------------------
# Snapped observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnappedPoints:
    """Where each observation lies on the road graph, in input row order.

    Observation ``k`` lies on the arc from vertex ``u[k]`` to vertex ``v[k]``
    (``u[k] <= v[k]``; ``u[k] == v[k]`` for a zero-length arc), at distance
    ``offset[k]`` from ``u[k]`` along an arc of length ``arc_length[k]``.
    """

    u: np.ndarray
    v: np.ndarray
    offset: np.ndarray
    arc_length: np.ndarray
    snap_distance: np.ndarray
    snapped_xy: np.ndarray

    def __len__(self) -> int:
        return len(self.u)

    def subset(self, index) -> SnappedPoints:
        """The snaps of some observations (a slice or an index array), in that order."""
        return SnappedPoints(
            u=self.u[index],
            v=self.v[index],
            offset=self.offset[index],
            arc_length=self.arc_length[index],
            snap_distance=self.snap_distance[index],
            snapped_xy=self.snapped_xy[index],
        )


def _empty_snaps() -> SnappedPoints:
    ints = np.empty(0, dtype=np.int64)
    floats = np.empty(0, dtype=float)
    return SnappedPoints(ints, ints, floats, floats, floats, np.empty((0, 2), dtype=float))


def snap_points(graph: RoadGraph, points: gpd.GeoDataFrame, *, pattern_name: str) -> SnappedPoints:
    """Snap points to their nearest arc with ``spaghetti`` and read the result as arrays.

    ``spaghetti`` numbers the points of a GeoDataFrame 0, 1, ..., n - 1 in row
    order (no ``idvariable`` is passed), so its point number is the row. The
    point pattern is removed from the network afterwards, because it is not
    used again and would otherwise stay in memory.
    """
    n = len(points)
    if n == 0:
        return _empty_snaps()
    network = graph.network
    network.snapobservations(points.reset_index(drop=True), pattern_name, attribute=False)
    try:
        pattern = network.pointpatterns[pattern_name]
        arc = np.full((n, 2), -1, dtype=np.int64)
        for arc_key, observations in pattern.obs_to_arc.items():
            for k in observations:
                arc[int(k)] = arc_key
        if (arc < 0).any():
            missing = np.flatnonzero(arc[:, 0] < 0)[:10].tolist()
            raise RuntimeError(f"spaghetti did not snap point rows {missing}")

        u, v = arc.min(axis=1), arc.max(axis=1)
        length = np.hypot(*(graph.vertex_xy[v] - graph.vertex_xy[u]).T)
        raw_offset = np.array([pattern.dist_to_vertex[k][int(u[k])] for k in range(n)], dtype=float)
        snap = np.array([pattern.dist_snapped[k] for k in range(n)], dtype=float)
        snapped_xy = np.array([pattern.snapped_coordinates[k] for k in range(n)], dtype=float)
    finally:
        network.pointpatterns.pop(pattern_name, None)

    if not (np.isfinite(raw_offset).all() and np.isfinite(snap).all() and (snap >= 0).all()):
        raise RuntimeError("spaghetti returned an invalid snap distance or offset")
    # An offset is measured on this same arc, so it can pass the arc end by
    # rounding error only. A clearly larger value means a wrong snap record.
    if np.any(raw_offset > length + 1e-6 * np.maximum(1.0, length)):
        raise RuntimeError("spaghetti returned a snap offset beyond the end of its arc")
    offset = np.clip(raw_offset, 0.0, length)
    # A point snapped onto a road vertex has exactly that vertex's coordinates
    # (spaghetti returns the vertex itself), so its offset is set to exactly 0
    # or the arc length. spaghetti measures the offset with ``math.hypot``,
    # while ``length`` uses ``np.hypot``; the two can differ in the last bit
    # (about 0.3% of random roads), and ``distinct_positions`` recognises a
    # vertex only by an exact offset.
    at_v = np.all(snapped_xy == graph.vertex_xy[v], axis=1)
    at_u = np.all(snapped_xy == graph.vertex_xy[u], axis=1)
    offset[at_v] = length[at_v]
    offset[at_u] = 0.0
    return SnappedPoints(
        u=u,
        v=v,
        offset=offset,
        arc_length=length,
        snap_distance=snap,
        snapped_xy=snapped_xy,
    )




def distinct_positions(snapped: SnappedPoints) -> tuple[np.ndarray, np.ndarray]:
    """Group observations snapped to exactly the same network position.

    Returns ``position`` (the position number of each observation) and
    ``representative`` (the first observation of each position). Positions are
    numbered in order of their first observation, so when the observations are
    in canonical order, the positions are too.

    An interior position is identified by ``(arc_u, arc_v, offset)``. An exact
    arc endpoint is identified instead by its network vertex ID, so the same
    road vertex reached through two incident arcs is one position. This does
    *not* merge geometrically coincident interiors of different arcs: those can
    represent a grade-separated or otherwise disconnected crossing.

    Observations at one position have road distance 0 to each other and the
    same distance to every other observation, so DBSCAN can treat them as one weighted point
    (``clustering.cluster_precomputed(..., position=...)``). Without this, k
    observations at one point form k(k - 1)/2 pairs.
    """
    n = len(snapped)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    at_u = (snapped.u == snapped.v) | (snapped.offset == 0.0)
    at_v = ~at_u & (snapped.offset == snapped.arc_length)
    is_vertex = at_u | at_v
    vertex = np.where(at_u, snapped.u, snapped.v)

    # kind 0 = vertex, kind 1 = arc interior. Values that do not participate
    # in a kind are filled with harmless sentinels; ``kind`` keeps the two
    # namespaces disjoint.
    kind = (~is_vertex).astype(np.int8)
    a = np.where(is_vertex, vertex, snapped.u)
    b = np.where(is_vertex, -1, snapped.v)
    offset_key = np.where(is_vertex, 0.0, snapped.offset)
    order = np.lexsort((offset_key, b, a, kind))
    kind_s, a_s, b_s, offset_s = kind[order], a[order], b[order], offset_key[order]
    new = np.ones(n, dtype=bool)
    new[1:] = (
        (kind_s[1:] != kind_s[:-1])
        | (a_s[1:] != a_s[:-1])
        | (b_s[1:] != b_s[:-1])
        | (offset_s[1:] != offset_s[:-1])
    )
    group = np.empty(n, dtype=np.int64)
    group[order] = np.cumsum(new) - 1
    n_groups = int(new.sum())
    first = np.full(n_groups, n, dtype=np.int64)
    np.minimum.at(first, group, np.arange(n))
    rank = np.argsort(first, kind="stable")
    renumber = np.empty(n_groups, dtype=np.int64)
    renumber[rank] = np.arange(n_groups)
    return renumber[group], first[rank]


# ---------------------------------------------------------------------------
# Pairs within eps
# ---------------------------------------------------------------------------


def _check_eps(eps: float) -> float:
    eps = float(eps)
    if not (np.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be a positive finite number, got {eps!r}")
    return eps


def _check_max_pairs(max_pairs: int) -> int:
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)) or max_pairs < 1:
        raise ValueError(f"max_neighbor_pairs must be a positive integer, got {max_pairs!r}")
    return int(max_pairs)


def _too_many_pairs(count: int, max_pairs: int, eps: float) -> ValueError:
    return ValueError(
        f"at least {count:,} pairs of snapped points are within cluster_eps={eps:g}, "
        f"above max_neighbor_pairs={max_pairs:,} (observations at one snapped position count as one point). "
        "Use a smaller cluster_eps or raise max_neighbor_pairs (each pair needs about 160 bytes at peak)."
    )


def _chunks(weights: np.ndarray, limit: int):
    """Consecutive index ranges whose weights sum to at most ``limit``.

    An index whose own weight is larger than ``limit`` forms a range alone.
    Used to bound the size of temporary arrays.
    """
    cum = np.concatenate([[0], np.cumsum(weights, dtype=np.int64)])
    first, n = 0, len(weights)
    while first < n:
        last = int(np.searchsorted(cum, cum[first] + limit, side="right")) - 1
        last = min(max(last, first + 1), n)
        yield slice(first, last)
        first = last


def _attachment(graph: RoadGraph, snapped: SnappedPoints):
    """For each road vertex, the snapped points on arcs that end there, nearest first.

    Returns CSR-style arrays: the points at vertex ``y`` are
    ``point[start[y]:start[y + 1]]``, at distances along their arcs
    ``along[start[y]:start[y + 1]]`` in ascending order. A point on a
    zero-length arc is attached once, at distance 0.
    """
    n = len(snapped)
    ends = np.concatenate([snapped.u, snapped.v])
    point = np.concatenate([np.arange(n), np.arange(n)])
    along = np.concatenate([snapped.offset, snapped.arc_length - snapped.offset])
    keep = np.ones(2 * n, dtype=bool)
    keep[n:] = snapped.u != snapped.v
    ends, point, along = ends[keep], point[keep], along[keep]
    order = np.lexsort((point, along, ends))
    ends, point, along = ends[order], point[order], along[order]
    start = np.searchsorted(ends, np.arange(graph.n_vertices + 1))
    return start, point, along


def _prefix_within(along: np.ndarray, lo: np.ndarray, hi: np.ndarray, reached: np.ndarray, eps: float) -> np.ndarray:
    """For each entry, how many of ``along[lo:hi]`` satisfy ``reached + along <= eps``.

    ``along`` is ascending on each range, so the test is true on a prefix of
    the range and a binary search finds its length. The test is the same
    floating-point sum that gives the stored distance, so the boundary at
    ``eps`` is exact.
    """
    first, lo, hi = lo.copy(), lo.copy(), hi.copy()
    while True:
        active = lo < hi
        if not active.any():
            return lo - first
        mid = (lo + hi) // 2
        ok = active & (reached + along[np.where(active, mid, 0)] <= eps)
        lo = np.where(ok, mid + 1, lo)
        hi = np.where(active & ~ok, mid, hi)


def _batch_size(n_vertices: int, n: int) -> int:
    """Searches per batch, so that one dense batch result stays within ``_SEARCH_BATCH_BYTES``.

    A batch of ``b`` searches returns ``b * (n_vertices + b + 1)`` float64 values.
    """
    budget = _SEARCH_BATCH_BYTES // 8
    nodes = n_vertices + 1
    b = int((-nodes + np.sqrt(float(nodes) ** 2 + 4.0 * budget)) // 2)
    return max(1, min(n, b))


def _search_graph(graph: RoadGraph, slots: int) -> tuple[csr_matrix, int]:
    """The road adjacency plus ``slots`` source nodes and one dead-end node.

    Source node ``V + k`` has exactly two outgoing edges; ``_set_sources``
    rewrites them in place for every batch, so the graph is built once. Road
    vertices have no edges to the added nodes, so the added nodes change no
    distance between road vertices. The dead-end node has no edges; it takes
    the second edge of a point on a zero-length arc, so that every row stays
    sorted and holds at most one entry per column. (SciPy's ``dijkstra`` does
    not modify the arrays of the graph it is given; a test checks the results
    against a freshly built graph.)
    """
    adjacency = graph.adjacency.copy()
    adjacency.sum_duplicates()
    adjacency.sort_indices()
    n_edges = adjacency.nnz
    n_nodes = graph.n_vertices + slots + 1
    indptr = np.concatenate([adjacency.indptr, n_edges + 2 * np.arange(1, slots + 1), [n_edges + 2 * slots]])
    indices = np.concatenate([adjacency.indices, np.zeros(2 * slots, dtype=adjacency.indices.dtype)])
    data = np.concatenate([adjacency.data, np.zeros(2 * slots)])
    return csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes)), n_edges


def _set_sources(search: csr_matrix, n_edges: int, snapped: SnappedPoints, start: int, stop: int, dead_end: int) -> None:
    """Point source node ``k`` at the two ends of the arc of point ``start + k``."""
    u, v = snapped.u[start:stop], snapped.v[start:stop]
    to_u = snapped.offset[start:stop]
    to_v = snapped.arc_length[start:stop] - to_u
    zero = u == v
    end = n_edges + 2 * (stop - start)
    search.indices[n_edges:end:2] = u
    search.indices[n_edges + 1 : end : 2] = np.where(zero, dead_end, v)
    search.data[n_edges:end:2] = to_u
    search.data[n_edges + 1 : end : 2] = np.where(zero, 0.0, to_v)


def _oversized_search_pairs(
    *,
    source_index: int,
    n_points: int,
    lo: np.ndarray,
    count: np.ndarray,
    reached: np.ndarray,
    attached: np.ndarray,
    along: np.ndarray,
    found_before: int,
    max_pairs: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate one unusually dense source search without a large candidate array.

    Normally ``_pairs_through_vertices`` handles several searches at once and
    materializes at most ``_CANDIDATE_CHUNK`` endpoint candidates.  If one
    source alone has more candidates than that, splitting those endpoint
    candidates naively could emit the same target twice (once through each end
    of its arc).  This fallback processes bounded slices, keeps the minimum
    distance for each target, and stops as soon as the unique-pair limit is
    exceeded.
    """
    best = np.full(n_points, np.inf, dtype=float)
    touched_parts: list[np.ndarray] = []
    n_touched = 0

    for entry in range(len(count)):
        first = int(lo[entry])
        stop = first + int(count[entry])
        for p0 in range(first, stop, _CANDIDATE_CHUNK):
            p1 = min(stop, p0 + _CANDIDATE_CHUNK)
            j = attached[p0:p1]
            keep = j > source_index
            if not keep.any():
                continue
            j = j[keep]
            d = reached[entry] + along[p0:p1][keep]

            unique_j = np.unique(j)
            new_j = unique_j[np.isinf(best[unique_j])]
            if len(new_j):
                n_touched += len(new_j)
                if found_before + n_touched > max_pairs:
                    raise _too_many_pairs(found_before + n_touched, max_pairs, eps)
                touched_parts.append(new_j.astype(_INDEX, copy=False))
            np.minimum.at(best, j, d)

    if not touched_parts:
        empty = np.empty(0, dtype=_INDEX)
        return empty, empty, np.empty(0, dtype=float)
    target = np.concatenate(touched_parts).astype(_INDEX, copy=False)
    distance = best[target].copy()
    source = np.full(len(target), source_index, dtype=_INDEX)
    return source, target, distance


def _pairs_through_vertices(graph: RoadGraph, snapped: SnappedPoints, eps: float, max_pairs: int):
    """Pairs whose shortest road path leaves the first arc through one of its ends.

    One bounded search (SciPy Dijkstra) runs from each point ``i``: from a
    source node with edges to the two ends of ``i``'s arc, weighted by the
    distances along the arc. The search stops at ``eps`` (SciPy keeps nodes at
    exactly ``eps``). For each road vertex ``y`` that it reaches at distance
    ``D``, every point ``j`` on an arc that ends at ``y``, at distance ``a``
    along that arc, gives the candidate distance ``D + a``; the pair keeps the
    smaller candidate of ``j``'s two ends. This is the formula ``spaghetti``
    uses for points on different arcs:

        min over (x, y) of  along(i, x) + D(x, y) + along(y, j).

    Only candidates with ``D + a <= eps`` are formed: the points at each vertex
    are sorted by ``a``, and a binary search finds them. Each pair is kept from
    the search of its lower index only, so it is produced once.

    Cost: each search returns one value per road vertex (SciPy's result is
    dense), so time grows with (points x road vertices). The graph holds no
    node per point, and each search meets only the points at the vertices it
    reaches, so no term grows with the square of the number of points.
    """
    n, n_vertices = len(snapped), graph.n_vertices
    start_of, attached, along = _attachment(graph, snapped)
    slots = _batch_size(n_vertices, n)
    search, n_edges = _search_graph(graph, slots)
    dead_end = n_vertices + slots
    found = 0
    for start in range(0, n, slots):
        stop = min(n, start + slots)
        _set_sources(search, n_edges, snapped, start, stop, dead_end)
        distance = dijkstra(search, directed=True, indices=n_vertices + np.arange(stop - start), limit=eps)
        within = distance[:, :n_vertices] <= eps
        for rows in _chunks(within.sum(axis=1), _REACHED_CHUNK):
            source, vertex = np.nonzero(within[rows])
            source += rows.start
            reached = distance[source, vertex]
            lo = start_of[vertex]
            count = _prefix_within(along, lo, start_of[vertex + 1], reached, eps)
            # Group the reached entries by search, so that both candidates of a
            # pair (one per end of j's arc) are compared in the same step.
            edges = np.searchsorted(source, np.arange(rows.start, rows.stop + 1))
            before = np.concatenate([[0], np.cumsum(count, dtype=np.int64)])
            per_search = before[edges[1:]] - before[edges[:-1]]
            for group in _chunks(per_search, _CANDIDATE_CHUNK):
                t0, t1 = int(edges[group.start]), int(edges[group.stop])
                c = count[t0:t1]
                total = int(c.sum())
                if total == 0:
                    continue
                if total > _CANDIDATE_CHUNK:
                    # ``_chunks`` only emits an overweight range when one
                    # source by itself exceeds the limit.  Process that source
                    # in bounded slices while deduplicating its two endpoint
                    # candidates before anything is yielded.
                    if group.stop - group.start != 1:
                        raise RuntimeError("candidate chunking produced an oversized multi-source group")
                    i_abs = start + rows.start + group.start
                    i, j, d = _oversized_search_pairs(
                        source_index=i_abs,
                        n_points=n,
                        lo=lo[t0:t1],
                        count=c,
                        reached=reached[t0:t1],
                        attached=attached,
                        along=along,
                        found_before=found,
                        max_pairs=max_pairs,
                        eps=eps,
                    )
                    if len(i):
                        found += len(i)
                        yield i, j, d
                    continue
                entry = np.repeat(np.arange(t0, t1), c)
                position = lo[entry] + (np.arange(total) - np.repeat(np.cumsum(c) - c, c))
                i = start + source[entry]
                j = attached[position]
                d = reached[entry] + along[position]
                keep = j > i
                i, j, d = i[keep], j[keep], d[keep]
                if len(i) == 0:
                    continue
                key = i.astype(np.int64) * np.int64(n) + j
                order = np.lexsort((d, key))
                key, i, j, d = key[order], i[order], j[order], d[order]
                first = np.ones(len(key), dtype=bool)
                first[1:] = key[1:] != key[:-1]
                found += int(first.sum())
                if found > max_pairs:
                    raise _too_many_pairs(found, max_pairs, eps)
                yield i[first].astype(_INDEX), j[first].astype(_INDEX), d[first]
        del distance, within


def _reach_on_arc(offset: np.ndarray, stop: np.ndarray, eps: float) -> np.ndarray:
    """For each sorted point ``k``, the first index in ``(k, stop[k])`` beyond ``eps``, or ``stop[k]``.

    ``offset`` is ascending within each arc, so ``offset[q] - offset[k] <= eps``
    is true on a prefix of the later points on the arc; a binary search finds
    its end. The test is the same floating-point subtraction that gives the
    stored distance, so the boundary at ``eps`` is exact.
    """
    k = np.arange(len(offset))
    base = offset
    lo, hi = k + 1, stop.copy()
    while True:
        active = lo < hi
        if not active.any():
            return lo
        mid = (lo + hi) // 2
        ok = active & (offset[np.where(active, mid, 0)] - base <= eps)
        lo = np.where(ok, mid + 1, lo)
        hi = np.where(active & ~ok, mid, hi)


def _pairs_on_same_arc(
    snapped: SnappedPoints,
    eps: float,
    max_pairs: int,
    *,
    batch_pairs: int = _SAME_ARC_BATCH_PAIRS,
):
    """Pairs on the same arc, measured directly along the arc.

    An arc is a straight segment, so no path that leaves the arc is shorter
    than the direct distance between two points on it. ``spaghetti`` uses the
    same rule for points on one arc. This case must be found separately: the
    search above only follows paths that leave the arc.

    The points are sorted by arc and offset, and for each point a binary search
    counts the later points on its arc within ``eps`` (``_reach_on_arc``). The
    count is exact, and it is checked against ``max_pairs`` before any pair
    array is created: same-arc pairs are distinct, so more of them than
    ``max_pairs`` means the whole graph has more too. The pairs are then
    produced in batches of at most ``batch_pairs``.
    """
    if batch_pairs < 1:
        raise ValueError("batch_pairs must be positive")
    n = len(snapped)
    if n < 2:
        return
    order = np.lexsort((snapped.offset, snapped.v, snapped.u))
    u, v, offset = snapped.u[order], snapped.v[order], snapped.offset[order]
    new_arc = np.ones(n, dtype=bool)
    new_arc[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
    arc_start = np.flatnonzero(new_arc)
    arc_stop = np.append(arc_start[1:], n)
    reach = _reach_on_arc(offset, np.repeat(arc_stop, arc_stop - arc_start), eps)
    ahead = reach - np.arange(n) - 1
    total = int(ahead.sum(dtype=np.int64))
    if total > max_pairs:
        raise _too_many_pairs(total, max_pairs, eps)

    for rows in _chunks(ahead, batch_pairs):
        if rows.stop - rows.start == 1 and ahead[rows.start] > batch_pairs:
            p = rows.start  # one point with more partners than a batch: split its row
            for q0 in range(p + 1, int(reach[p]), batch_pairs):
                q = np.arange(q0, min(int(reach[p]), q0 + batch_pairs))
                yield np.full(len(q), order[p], dtype=_INDEX), order[q].astype(_INDEX), offset[q] - offset[p]
            continue
        c = ahead[rows]
        total = int(c.sum())
        if total == 0:
            continue
        first = np.repeat(np.arange(rows.start, rows.stop), c)
        second = first + 1 + (np.arange(total) - np.repeat(np.cumsum(c) - c, c))
        yield order[first].astype(_INDEX), order[second].astype(_INDEX), offset[second] - offset[first]


def neighbor_graph(
    graph: RoadGraph,
    snapped: SnappedPoints,
    *,
    eps: float,
    max_pairs: int = DEFAULT_MAX_NEIGHBOR_PAIRS,
) -> csr_matrix:
    """All pairs of snapped points with road distance ``<= eps``, as a sparse ``(n, n)`` matrix.

    The caller passes one point per distinct snapped position
    (``distinct_positions``), so ``max_pairs`` counts pairs of positions.
    Stored entries are exactly the pairs ``(i, j)``, ``i != j``, with
    ``d_R(i, j) <= eps``; the matrix is symmetric. A distance of zero (two
    observations snapped to the same position) is stored as an explicit zero:
    for scikit-learn a stored zero is a neighbour, while a missing entry is
    not. Pairs farther apart, including pairs on disconnected road
    components, are simply absent, so no stand-in value for infinity is needed.
    """
    eps = _check_eps(eps)
    max_pairs = _check_max_pairs(max_pairs)
    n = len(snapped)
    if n < 2:
        return csr_matrix((n, n), dtype=float)

    if n >= 2**31:
        raise ValueError("more than 2**31 - 1 observations in one clustering run are not supported")
    through_parts = list(_pairs_through_vertices(graph, snapped, eps, max_pairs))
    if through_parts:
        first = np.concatenate([p[0] for p in through_parts])
        second = np.concatenate([p[1] for p in through_parts])
        distance = np.concatenate([p[2] for p in through_parts]).astype(float)
        low, high = np.minimum(first, second), np.maximum(first, second)
        del first, second, through_parts

        # _pairs_through_vertices emits each unordered pair once, but sorting
        # gives us a compact key table for merging direct same-arc distances.
        key = low.astype(np.int64) * np.int64(n) + high.astype(np.int64)
        order = np.argsort(key, kind="stable")
        low, high, distance, key = low[order], high[order], distance[order], key[order]
    else:
        low = np.empty(0, dtype=_INDEX)
        high = np.empty(0, dtype=_INDEX)
        distance = np.empty(0, dtype=float)
        key = np.empty(0, dtype=np.int64)

    # Merge same-arc pairs incrementally.  A direct same-arc pair can also be
    # found through the arc ends (or another cycle).  For such an overlap keep
    # the smaller distance; otherwise append a new unordered pair.  This keeps
    # the stored unique pair set at or below max_pairs throughout the merge.
    added_low: list[np.ndarray] = []
    added_high: list[np.ndarray] = []
    added_distance: list[np.ndarray] = []
    n_unique = len(low)
    for first, second, direct in _pairs_on_same_arc(snapped, eps, max_pairs):
        direct_low = np.minimum(first, second).astype(_INDEX, copy=False)
        direct_high = np.maximum(first, second).astype(_INDEX, copy=False)
        direct_key = direct_low.astype(np.int64) * np.int64(n) + direct_high.astype(np.int64)

        if len(key):
            pos = np.searchsorted(key, direct_key)
            overlap = pos < len(key)
            overlap[overlap] &= key[pos[overlap]] == direct_key[overlap]
            if overlap.any():
                np.minimum.at(distance, pos[overlap], direct[overlap])
        else:
            overlap = np.zeros(len(direct_key), dtype=bool)

        new = ~overlap
        n_new = int(new.sum())
        if n_unique + n_new > max_pairs:
            raise _too_many_pairs(n_unique + n_new, max_pairs, eps)
        if n_new:
            added_low.append(direct_low[new])
            added_high.append(direct_high[new])
            added_distance.append(direct[new])
            n_unique += n_new

    if added_low:
        low = np.concatenate([low, *added_low])
        high = np.concatenate([high, *added_high])
        distance = np.concatenate([distance, *added_distance])

    matrix = coo_matrix(
        (np.concatenate([distance, distance]), (np.concatenate([low, high]), np.concatenate([high, low]))),
        shape=(n, n),
    ).tocsr()
    if matrix.nnz != 2 * len(low):  # explicit zeros must survive the conversion
        raise RuntimeError("sparse conversion dropped stored neighbour pairs")
    return matrix
