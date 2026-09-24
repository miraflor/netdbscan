import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from netdbscan import NetDBSCANConfig, cluster_geodataframes

CRS = "EPSG:32651"


def _data():
    network = gpd.GeoDataFrame(geometry=[LineString([(0,0),(100,0)])], crs=CRS)
    boundary = gpd.GeoDataFrame(geometry=[Polygon([(-10,-10),(110,-10),(110,10),(-10,10)])], crs=CRS)
    points = gpd.GeoDataFrame(
        {"point_id":["a","b","c","outside"]},
        geometry=[Point(10,1), Point(15,-1), Point(90,1), Point(200,0)], crs=CRS
    )
    return points,boundary,network


def test_empty_after_boundary_returns_empty_without_spaghetti():
    network = gpd.GeoDataFrame(geometry=[LineString([(0,0),(100,0)])], crs=CRS)
    boundary = gpd.GeoDataFrame(geometry=[Polygon([(0,0),(10,0),(10,10),(0,10)])], crs=CRS)
    points = gpd.GeoDataFrame({"point_id":["x"]}, geometry=[Point(50,50)], crs=CRS)
    result = cluster_geodataframes(points=points,boundary=boundary,network=network,config=NetDBSCANConfig(eps=10))
    assert result.empty
    assert "cluster_id" in result


def test_small_end_to_end():
    pytest.importorskip("spaghetti")
    points,boundary,network=_data()
    r=cluster_geodataframes(points=points,boundary=boundary,network=network,config=NetDBSCANConfig(eps=10,min_samples=2))
    assert r["point_id"].tolist() == ["a","b","c"]
    assert r.loc[r.point_id.isin(["a","b"]),"cluster_id"].notna().all()
    assert r.loc[r.point_id.eq("c"),"is_noise"].item()
