from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from netdbscan import NetDBSCANConfig, cluster_files_by_column

CRS = "EPSG:32651"


def _frames():
    points = gpd.GeoDataFrame(
        {
            "point_id": ["a", "b", "c", "d"],
            "kind": ["03", "", pd.NA, "03"],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)],
        crs=CRS,
    )
    boundary = gpd.GeoDataFrame(
        geometry=[Polygon([(-1, -1), (4, -1), (4, 1), (-1, 1)])],
        crs=CRS,
    )
    network = gpd.GeoDataFrame(
        geometry=[LineString([(-1, 0), (4, 0)])],
        crs=CRS,
    )
    return points, boundary, network


def test_batch_by_column_includes_blank_and_null(monkeypatch, tmp_path):
    points, boundary, network = _frames()

    def fake_read_vector(path, *, name, layer=None):
        return {"points": points, "boundary": boundary, "network": network}[name].copy()

    seen_groups = []

    def fake_cluster_geodataframes(*, points, boundary, network, config, point_id_col):
        seen_groups.append(points["kind"].tolist())
        result = points.copy()
        result["cluster_id"] = None
        result["is_noise"] = True
        result["is_core"] = False
        result["snap_distance"] = 0.0
        result["snapped_x"] = result.geometry.x
        result["snapped_y"] = result.geometry.y
        return result

    written = []

    def fake_write(path, frame, *, force=False):
        written.append(Path(path))
        return Path(path)

    monkeypatch.setattr("netdbscan.pipeline.read_vector", fake_read_vector)
    monkeypatch.setattr("netdbscan.pipeline.cluster_geodataframes", fake_cluster_geodataframes)
    monkeypatch.setattr("netdbscan.pipeline.write_geoparquet", fake_write)

    outputs = cluster_files_by_column(
        points_path="points.parquet",
        boundary_path="boundary.parquet",
        network_path="roads.parquet",
        output_dir=tmp_path,
        group_col="kind",
        point_id_col="point_id",
        config=NetDBSCANConfig(eps=10),
    )

    assert {path.name for path in outputs} == {
        "group_03.parquet",
        "group___blank__.parquet",
        "group___null__.parquet",
    }
    assert sorted(len(group) for group in seen_groups) == [1, 1, 2]
    assert outputs == written


def test_batch_missing_group_column_is_clear(monkeypatch, tmp_path):
    points, boundary, network = _frames()

    def fake_read_vector(path, *, name, layer=None):
        return {"points": points, "boundary": boundary, "network": network}[name].copy()

    monkeypatch.setattr("netdbscan.pipeline.read_vector", fake_read_vector)

    with pytest.raises(ValueError, match="group column 'missing' not found"):
        cluster_files_by_column(
            points_path="points.parquet",
            boundary_path="boundary.parquet",
            network_path="roads.parquet",
            output_dir=tmp_path,
            group_col="missing",
            config=NetDBSCANConfig(eps=10),
        )
