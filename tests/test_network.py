import numpy as np
from scipy.sparse import csr_matrix

from netdbscan.network import RoadGraph, SnappedPoints, distinct_positions, neighbor_graph


def line_graph():
    xy = np.array([[0.,0.],[10.,0.],[20.,0.]])
    adj = csr_matrix(np.array([[0.,10.,0.],[10.,0.,10.],[0.,10.,0.]]))
    return RoadGraph(None, xy, np.array([0,1]), np.array([1,2]), np.array([10.,10.]), adj, np.array([0,0,0]))


def snaps(u,v,offset,length):
    u=np.asarray(u); v=np.asarray(v); offset=np.asarray(offset,dtype=float); length=np.asarray(length,dtype=float)
    xy=np.zeros((len(u),2),dtype=float)
    return SnappedPoints(u,v,offset,length,np.zeros(len(u)),xy)


def test_same_arc_distance():
    g=line_graph(); s=snaps([0,0],[1,1],[2,8],[10,10])
    m=neighbor_graph(g,s,eps=6,max_pairs=10)
    assert m[0,1] == 6


def test_distance_across_adjacent_arcs():
    g=line_graph(); s=snaps([0,1],[1,2],[8,3],[10,10])
    m=neighbor_graph(g,s,eps=5,max_pairs=10)
    assert m[0,1] == 5


def test_pair_outside_eps_absent():
    g=line_graph(); s=snaps([0,1],[1,2],[2,8],[10,10])
    m=neighbor_graph(g,s,eps=5,max_pairs=10)
    assert m.nnz == 0


def test_endpoint_positions_on_incident_arcs_are_one_position():
    s=snaps([0,1],[1,2],[10,0],[10,10])
    pos, rep=distinct_positions(s)
    assert pos.tolist() == [0,0]
    assert rep.tolist() == [0]


def test_interiors_on_different_arcs_are_not_merged():
    s=snaps([0,1],[1,2],[5,5],[10,10])
    pos, rep=distinct_positions(s)
    assert pos.tolist() == [0,1]
