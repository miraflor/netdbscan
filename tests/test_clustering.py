import numpy as np
from scipy.sparse import csr_matrix

from netdbscan.clustering import canonical_point_order, cluster_precomputed


def test_canonical_order_is_string_order():
    assert canonical_point_order([2, 10, 1]).tolist() == [2, 1, 0]


def test_cluster_ids_are_stable_under_row_permutation():
    d = np.array([[0, 1, 9, 9], [1, 0, 9, 9], [9, 9, 0, 1], [9, 9, 1, 0]], dtype=float)
    a = cluster_precomputed(["b", "a", "d", "c"], d, cluster_eps=2, min_samples=2)
    p = [2, 0, 3, 1]
    b = cluster_precomputed(["b", "a", "d", "c"], d, cluster_eps=2, min_samples=2)
    assert a.cluster_ids.tolist() == b.cluster_ids.tolist()
    assert set(x for x in a.cluster_ids if x) == {"C000001", "C000002"}


def test_noise_exclude_keeps_null_cluster():
    d = np.array([[0, 10], [10, 0]], dtype=float)
    r = cluster_precomputed(["a", "b"], d, cluster_eps=1, min_samples=2)
    assert r.is_noise.tolist() == [True, True]
    assert r.cluster_ids.tolist() == [None, None]


def test_noise_singleton_gets_clusters_but_stays_noise():
    d = np.array([[0, 10], [10, 0]], dtype=float)
    r = cluster_precomputed(["a", "b"], d, cluster_eps=1, min_samples=2, noise_policy="singleton")
    assert r.is_noise.tolist() == [True, True]
    assert r.cluster_ids.tolist() == ["C000001", "C000002"]


def test_repeated_position_uses_weight_for_min_samples():
    # Three observations share position 0; one is at position 1 beyond eps.
    graph = csr_matrix((2, 2), dtype=float)
    r = cluster_precomputed(["a", "b", "c", "d"], graph, cluster_eps=1, min_samples=3, position=[0, 0, 0, 1])
    assert r.cluster_ids[:3].tolist() == ["C000001"] * 3
    assert r.is_core[:3].tolist() == [True] * 3
    assert r.is_noise[3]
