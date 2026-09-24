"""Vector input validation, CRS harmonization, boundary filtering, and output."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

PARQUET_SUFFIXES = {".parquet", ".geoparquet"}
FILE_VECTOR_SUFFIXES = {".shp", ".gpkg"}
INPUT_SUFFIXES = {*FILE_VECTOR_SUFFIXES, *PARQUET_SUFFIXES}
OUTPUT_COLUMNS = {
    "cluster_id",
    "is_noise",
    "is_core",
    "snap_distance",
    "snapped_x",
    "snapped_y",
}


def read_vector(
    path: str | Path,
    *,
    name: str,
    layer: str | None = None,
) -> gpd.GeoDataFrame:
    """Read a GeoParquet, Shapefile, or GeoPackage layer.

    GeoPackages may contain several layers. If ``layer`` is omitted, a
    one-layer GeoPackage is accepted automatically; a multi-layer GeoPackage
    is rejected so the caller must choose explicitly rather than silently
    receiving an arbitrary layer.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in INPUT_SUFFIXES:
        choices = ", ".join(sorted(INPUT_SUFFIXES))
        raise ValueError(f"{name} must use one of: {choices}")

    if suffix in PARQUET_SUFFIXES:
        if layer is not None:
            raise ValueError(f"{name} layer may be specified only for .gpkg inputs")
        return gpd.read_parquet(path)

    if suffix == ".gpkg":
        layers = gpd.list_layers(path)
        names = layers["name"].astype(str).tolist()
        if layer is None:
            if len(names) != 1:
                choices = ", ".join(repr(x) for x in names)
                raise ValueError(
                    f"{name} GeoPackage contains {len(names)} layers ({choices}); "
                    f"specify which layer to read"
                )
            layer = names[0]
        elif layer not in names:
            choices = ", ".join(repr(x) for x in names)
            raise ValueError(
                f"{name} layer {layer!r} not found in {path}; available layers: {choices}"
            )
        return gpd.read_file(path, layer=layer)

    if layer is not None:
        raise ValueError(f"{name} layer may be specified only for .gpkg inputs")
    return gpd.read_file(path)


def prepare_network(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if network.empty:
        raise ValueError("network is empty")
    if network.crs is None:
        raise ValueError("network has no CRS")
    if not network.crs.is_projected:
        raise ValueError("network must use a projected CRS")
    bad = network.geometry.isna() | network.geometry.is_empty
    if bad.any():
        raise ValueError(f"network contains {int(bad.sum())} null/empty geometries")
    allowed = network.geometry.geom_type.isin(["LineString", "MultiLineString"])
    if not allowed.all():
        kinds = sorted(network.loc[~allowed].geometry.geom_type.unique().tolist())
        raise ValueError(f"network must contain only LineString/MultiLineString geometries; found {kinds}")
    if not network.geometry.is_valid.all():
        raise ValueError("network contains invalid geometry; repair it explicitly before analysis")

    # ``spaghetti`` should receive one LineString per row.  Passing a
    # MultiLineString through libpysal can connect the end of one part to the
    # start of the next part, creating a false arc.
    work = network.copy()
    if (work.geometry.geom_type == "MultiLineString").any():
        work = work.explode(index_parts=False, ignore_index=True)
    work = work.reset_index(drop=True)
    coords = shapely.get_coordinates(work.geometry.to_numpy())
    if not np.isfinite(coords).all():
        raise ValueError("network contains NaN or infinite coordinates")
    if (work.geometry.length <= 0).any():
        raise ValueError("network contains zero-length LineStrings")
    return work


def prepare_boundary(boundary: gpd.GeoDataFrame, analysis_crs) -> gpd.GeoDataFrame:
    if boundary.empty:
        raise ValueError("boundary is empty")
    if boundary.crs is None:
        raise ValueError("boundary has no CRS")
    bad = boundary.geometry.isna() | boundary.geometry.is_empty
    if bad.any():
        raise ValueError(f"boundary contains {int(bad.sum())} null/empty geometries")
    allowed = boundary.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if not allowed.all():
        kinds = sorted(boundary.loc[~allowed].geometry.geom_type.unique().tolist())
        raise ValueError(f"boundary must contain only Polygon/MultiPolygon geometries; found {kinds}")
    if not boundary.geometry.is_valid.all():
        raise ValueError("boundary contains invalid geometry; repair it explicitly before analysis")
    work = boundary.to_crs(analysis_crs) if boundary.crs != analysis_crs else boundary.copy()
    if not work.geometry.is_valid.all():
        raise ValueError("boundary became invalid after reprojection to the network CRS")
    geom = work.geometry.union_all()
    if geom.is_empty:
        raise ValueError("boundary union is empty")
    return gpd.GeoDataFrame({"geometry": [geom]}, geometry="geometry", crs=analysis_crs)


def prepare_points(points: gpd.GeoDataFrame, analysis_crs, point_id_col: str) -> gpd.GeoDataFrame:
    if points.crs is None:
        raise ValueError("points has no CRS")
    if point_id_col not in points.columns:
        raise ValueError(f"point id column {point_id_col!r} not found")
    if points[point_id_col].isna().any():
        raise ValueError("point_id values must not be null")
    if points[point_id_col].duplicated().any():
        raise ValueError("duplicate point_id values are not allowed")
    if points[point_id_col].map(str).duplicated().any():
        raise ValueError("point_id values must be unique after string conversion")
    kinds = {type(v.item() if isinstance(v, np.generic) else v) for v in points[point_id_col].tolist()}
    if len(kinds) > 1:
        names = sorted(k.__name__ for k in kinds)
        raise ValueError(f"point_id values mix Python types {names}; use one storage type")
    collisions = sorted(OUTPUT_COLUMNS.intersection(points.columns))
    if collisions:
        raise ValueError(f"points already contains reserved output columns: {collisions}")

    bad = points.geometry.isna() | points.geometry.is_empty
    if bad.any():
        raise ValueError(f"points contains {int(bad.sum())} null/empty geometries")
    if not points.geometry.geom_type.isin(["Point"]).all():
        raise ValueError("points must contain only Point geometries")
    work = points.to_crs(analysis_crs) if points.crs != analysis_crs else points.copy()
    work = work.reset_index(drop=True)
    coords = shapely.get_coordinates(work.geometry.to_numpy())
    if len(coords) and not np.isfinite(coords).all():
        raise ValueError("points contains NaN or infinite coordinates")
    return work


def clip_points(points: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Retain points covered by the boundary, including points on its edge."""
    geom = boundary.geometry.iloc[0]
    shapely.prepare(geom)
    keep = shapely.covers(geom, points.geometry.to_numpy())
    return points.loc[keep].copy().reset_index(drop=True)


def write_geoparquet(path: str | Path, frame: gpd.GeoDataFrame, *, force: bool = False) -> Path:
    """Atomically write the grouped point layer as GeoParquet."""
    path = Path(path)
    if path.suffix.lower() not in PARQUET_SUFFIXES:
        raise ValueError("output must end in .parquet or .geoparquet")
    if path.exists() and not force:
        raise FileExistsError(f"output already exists: {path}; pass force=True or --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
