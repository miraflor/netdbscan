"""High-level point -> boundary filter -> road-network DBSCAN pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import geopandas as gpd
import numpy as np
import pandas as pd

from .clustering import canonical_point_order, cluster_precomputed
from .io import (
    clip_points,
    prepare_boundary,
    prepare_network,
    prepare_points,
    read_vector,
    write_geoparquet,
)
from .network import DEFAULT_MAX_NEIGHBOR_PAIRS, build_road_graph, distinct_positions, neighbor_graph, snap_points

NoisePolicy = Literal["exclude", "singleton"]


@dataclass(frozen=True)
class NetDBSCANConfig:
    """Clustering parameters, all distances in the network CRS linear units."""

    eps: float
    min_samples: int = 5
    noise_policy: NoisePolicy = "exclude"
    max_snap_distance: float | None = None
    max_neighbor_pairs: int = DEFAULT_MAX_NEIGHBOR_PAIRS

    def __post_init__(self) -> None:
        if isinstance(self.eps, bool) or not (isinstance(self.eps, (int, float)) and math.isfinite(self.eps) and self.eps > 0):
            raise ValueError("eps must be a positive finite number")
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, (int, np.integer)) or self.min_samples < 1:
            raise ValueError("min_samples must be an integer >= 1")
        if self.noise_policy not in {"exclude", "singleton"}:
            raise ValueError("noise_policy must be 'exclude' or 'singleton'")
        if self.max_snap_distance is not None:
            value = self.max_snap_distance
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
                raise ValueError("max_snap_distance must be >= 0, inf, or None")
        if isinstance(self.max_neighbor_pairs, bool) or not isinstance(self.max_neighbor_pairs, (int, np.integer)) or self.max_neighbor_pairs < 1:
            raise ValueError("max_neighbor_pairs must be an integer >= 1")


def _check_snap_distance(point_ids, distances: np.ndarray, limit: float | None) -> None:
    if limit is None or math.isinf(limit):
        return
    too_far = distances > limit
    if too_far.any():
        k = int(np.flatnonzero(too_far)[0])
        raise ValueError(
            f"point {point_ids[k]!r} snapped {distances[k]:g} network-CRS units; "
            f"max_snap_distance={limit:g} ({int(too_far.sum())} point(s) exceed it)"
        )


def _prepare_inside_points(
    points: gpd.GeoDataFrame,
    *,
    analysis_crs,
    boundary_work: gpd.GeoDataFrame,
    point_id_col: str,
) -> gpd.GeoDataFrame:
    """Validate, reproject, clip, and deterministically order one point subset."""
    points_work = prepare_points(points, analysis_crs, point_id_col)
    inside = clip_points(points_work, boundary_work)
    if len(inside):
        order = canonical_point_order(inside[point_id_col].tolist())
        inside = inside.iloc[order].reset_index(drop=True)
    return inside


def _empty_cluster_result(inside: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the normal output schema for an empty boundary-covered subset."""
    result = inside.copy()
    result["cluster_id"] = np.asarray([], dtype=object)
    result["is_noise"] = np.asarray([], dtype=bool)
    result["is_core"] = np.asarray([], dtype=bool)
    result["snap_distance"] = np.asarray([], dtype=float)
    result["snapped_x"] = np.asarray([], dtype=float)
    result["snapped_y"] = np.asarray([], dtype=float)
    return result


def _cluster_prepared_points(
    *,
    inside: gpd.GeoDataFrame,
    graph,
    config: NetDBSCANConfig,
    point_id_col: str,
) -> gpd.GeoDataFrame:
    """Cluster a non-empty, already prepared point subset on a shared road graph."""
    snapped = snap_points(graph, inside, pattern_name="netdbscan_points")
    _check_snap_distance(inside[point_id_col].tolist(), snapped.snap_distance, config.max_snap_distance)

    position, representative = distinct_positions(snapped)
    representative_snaps = snapped.subset(representative)
    distances = neighbor_graph(
        graph,
        representative_snaps,
        eps=config.eps,
        max_pairs=config.max_neighbor_pairs,
    )
    clustered = cluster_precomputed(
        inside[point_id_col].tolist(),
        distances,
        cluster_eps=config.eps,
        min_samples=config.min_samples,
        noise_policy=config.noise_policy,
        position=position,
    )

    if [str(x) for x in inside[point_id_col].tolist()] != clustered.point_keys:
        raise RuntimeError("internal point ordering changed during clustering")
    result = inside.copy()
    result["cluster_id"] = clustered.cluster_ids
    result["is_noise"] = clustered.is_noise
    result["is_core"] = clustered.is_core
    result["snap_distance"] = snapped.snap_distance
    result["snapped_x"] = snapped.snapped_xy[:, 0]
    result["snapped_y"] = snapped.snapped_xy[:, 1]
    return result


def cluster_geodataframes(
    *,
    points: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    network: gpd.GeoDataFrame,
    config: NetDBSCANConfig,
    point_id_col: str = "point_id",
) -> gpd.GeoDataFrame:
    """Cluster boundary-covered points by shortest-path distance along ``network``.

    The returned geometry is the original point geometry reprojected to the
    projected network CRS. All original point attributes are preserved and six
    fields are appended: ``cluster_id``, ``is_noise``, ``is_core``,
    ``snap_distance``, ``snapped_x`` and ``snapped_y``.
    """
    if not isinstance(config, NetDBSCANConfig):
        raise TypeError("config must be a NetDBSCANConfig")

    network_work = prepare_network(network)
    boundary_work = prepare_boundary(boundary, network_work.crs)
    inside = _prepare_inside_points(
        points,
        analysis_crs=network_work.crs,
        boundary_work=boundary_work,
        point_id_col=point_id_col,
    )
    if not len(inside):
        return _empty_cluster_result(inside)

    graph = build_road_graph(network_work)
    return _cluster_prepared_points(
        inside=inside,
        graph=graph,
        config=config,
        point_id_col=point_id_col,
    )


def cluster_files(
    *,
    points_path: str | Path,
    boundary_path: str | Path,
    network_path: str | Path,
    output_path: str | Path,
    config: NetDBSCANConfig,
    point_id_col: str = "point_id",
    points_layer: str | None = None,
    boundary_layer: str | None = None,
    network_layer: str | None = None,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """File-based wrapper. Inputs may be GeoParquet, Shapefile, or GeoPackage."""
    result = cluster_geodataframes(
        points=read_vector(points_path, name="points", layer=points_layer),
        boundary=read_vector(boundary_path, name="boundary", layer=boundary_layer),
        network=read_vector(network_path, name="network", layer=network_layer),
        config=config,
        point_id_col=point_id_col,
    )
    write_geoparquet(output_path, result, force=force)
    return result

def _group_output_token(value) -> str:
    """Return a filesystem-safe token for one group value."""
    if pd.isna(value):
        return "__null__"
    text = str(value)
    if text == "":
        return "__blank__"
    return quote(text, safe="-_.~")


def cluster_files_by_column(
    *,
    points_path: str | Path,
    boundary_path: str | Path,
    network_path: str | Path,
    output_dir: str | Path,
    group_col: str,
    config: NetDBSCANConfig,
    point_id_col: str = "point_id",
    points_layer: str | None = None,
    boundary_layer: str | None = None,
    network_layer: str | None = None,
    force: bool = False,
) -> list[Path]:
    """Cluster each unique value of ``group_col`` independently.

    Inputs are read once. The network and boundary are prepared once, and the
    road graph is built at most once and reused across all non-empty groups.
    Each unique point-layer value is still validated, clipped, snapped, and
    clustered independently. Null and blank values are retained as groups.

    Cluster IDs restart independently within each group.
    """
    if not isinstance(config, NetDBSCANConfig):
        raise TypeError("config must be a NetDBSCANConfig")

    points = read_vector(points_path, name="points", layer=points_layer)
    boundary = read_vector(boundary_path, name="boundary", layer=boundary_layer)
    network = read_vector(network_path, name="network", layer=network_layer)

    if group_col not in points.columns:
        raise ValueError(f"group column {group_col!r} not found")

    values = list(pd.unique(points[group_col]))
    values.sort(
        key=lambda value: (
            1 if pd.isna(value) else 0,
            "" if pd.isna(value) else str(value),
        )
    )

    output_dir = Path(output_dir)
    planned: list[tuple[object, Path]] = []
    seen_names: dict[str, object] = {}
    for value in values:
        token = _group_output_token(value)
        filename = f"group_{token}.parquet"
        if filename in seen_names:
            other = seen_names[filename]
            raise ValueError(
                f"group values {other!r} and {value!r} map to the same output filename "
                f"{filename!r}; clean or recode {group_col!r}"
            )
        seen_names[filename] = value
        planned.append((value, output_dir / filename))

    # Refuse before any clustering so a pre-existing group output cannot leave
    # a partially updated folder.
    if not force:
        existing = [path for _, path in planned if path.exists()]
        if existing:
            raise FileExistsError(
                f"output already exists: {existing[0]}; pass force=True or --force"
            )

    network_work = prepare_network(network)
    boundary_work = prepare_boundary(boundary, network_work.crs)

    # Building the road graph is the expensive network-level setup. Delay it
    # until the first group with at least one boundary-covered point, then
    # reuse the same immutable graph for every remaining group.
    graph = None
    outputs: list[Path] = []
    for value, output_path in planned:
        if pd.isna(value):
            mask = points[group_col].isna()
        else:
            mask = points[group_col].eq(value)
        subset = points.loc[mask].copy().reset_index(drop=True)
        inside = _prepare_inside_points(
            subset,
            analysis_crs=network_work.crs,
            boundary_work=boundary_work,
            point_id_col=point_id_col,
        )
        if len(inside):
            if graph is None:
                graph = build_road_graph(network_work)
            result = _cluster_prepared_points(
                inside=inside,
                graph=graph,
                config=config,
                point_id_col=point_id_col,
            )
        else:
            result = _empty_cluster_result(inside)
        outputs.append(write_geoparquet(output_path, result, force=force))

    return outputs
