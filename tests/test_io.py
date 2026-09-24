import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from netdbscan.io import clip_points, prepare_boundary, prepare_network, prepare_points, read_vector

CRS = "EPSG:32651"


def test_boundary_covers_edge_point():
    boundary = gpd.GeoDataFrame(geometry=[Polygon([(0,0),(10,0),(10,10),(0,10)])], crs=CRS)
    points = gpd.GeoDataFrame({"point_id":["edge","out"]}, geometry=[Point(0,5), Point(-1,5)], crs=CRS)
    p = prepare_points(points, CRS, "point_id")
    b = prepare_boundary(boundary, CRS)
    got = clip_points(p, b)
    assert got["point_id"].tolist() == ["edge"]


def test_network_requires_projected_crs():
    roads = gpd.GeoDataFrame(geometry=[LineString([(0,0),(1,0)])], crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected"):
        prepare_network(roads)


def test_multiline_is_exploded():
    geom = MultiLineString([[(0,0),(1,0)], [(10,0),(11,0)]])
    roads = gpd.GeoDataFrame(geometry=[geom], crs=CRS)
    assert len(prepare_network(roads)) == 2


def test_duplicate_ids_refused():
    points = gpd.GeoDataFrame({"point_id":["x","x"]}, geometry=[Point(0,0),Point(1,0)], crs=CRS)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_points(points, CRS, "point_id")


def test_reserved_output_columns_refused():
    points = gpd.GeoDataFrame({"point_id":["x"], "cluster_id":["old"]}, geometry=[Point(0,0)], crs=CRS)
    with pytest.raises(ValueError, match="reserved"):
        prepare_points(points, CRS, "point_id")


def test_read_single_layer_geopackage(tmp_path):
    path = tmp_path / "points.gpkg"
    frame = gpd.GeoDataFrame(
        {"point_id": ["a", "b"]},
        geometry=[Point(0, 0), Point(1, 0)],
        crs=CRS,
    )
    frame.to_file(path, layer="observations", driver="GPKG")
    got = read_vector(path, name="points")
    assert got["point_id"].tolist() == ["a", "b"]


def test_multilayer_geopackage_requires_layer(tmp_path):
    path = tmp_path / "multi.gpkg"
    a = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs=CRS)
    b = gpd.GeoDataFrame({"id": [2]}, geometry=[Point(1, 0)], crs=CRS)
    a.to_file(path, layer="first", driver="GPKG")
    b.to_file(path, layer="second", driver="GPKG")

    with pytest.raises(ValueError, match="contains 2 layers"):
        read_vector(path, name="points")

    got = read_vector(path, name="points", layer="second")
    assert got["id"].tolist() == [2]


def test_missing_geopackage_layer_is_clear(tmp_path):
    path = tmp_path / "points.gpkg"
    frame = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs=CRS)
    frame.to_file(path, layer="observations", driver="GPKG")
    with pytest.raises(ValueError, match="available layers"):
        read_vector(path, name="points", layer="missing")


def test_layer_option_rejected_for_parquet(tmp_path):
    path = tmp_path / "points.parquet"
    with pytest.raises(ValueError, match="only for .gpkg"):
        read_vector(path, name="points", layer="anything")
