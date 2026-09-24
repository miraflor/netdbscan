# netdbscan

`netdbscan` clusters point observations with DBSCAN using **shortest-path distance along a supplied line network** rather than Euclidean distance.

The pipeline is deliberately small:

```text
point layer
    ↓
filter to polygon boundary
    ↓
snap retained points to the line network
    ↓
find sparse road-network neighbours within eps
    ↓
scikit-learn DBSCAN
    ↓
grouped point layer
```

The network is **not clipped** by the boundary. The boundary controls point eligibility only. A shortest path between two retained points may therefore leave the polygon when the supplied network does so.

## Input formats

**GeoParquet is the recommended format** for points, boundary and network. GeoPackage and Shapefile are accepted for interoperability.

GeoParquet is the better default for a Python workflow because it preserves modern column types and CRS metadata and avoids classic Shapefile restrictions such as 10-character field names, weak null handling and legacy text encoding. GeoPackage is a good choice when a GIS workflow benefits from a single portable container or named layers. The output is always GeoParquet.

Inputs:

- **points** — `Point` geometries with a unique `point_id` column (configurable);
- **boundary** — `Polygon` or `MultiPolygon` geometries;
- **network** — `LineString` or `MultiLineString` geometries in a **projected CRS**.

The network CRS is the analysis CRS. Points and boundary are reprojected to it. `eps` and `max_snap_distance` use that CRS's linear units; the package does not assume metres.

## Output

The output contains only points covered by the boundary, including points exactly on the boundary. All original point attributes are preserved. The following fields are appended:

| field | meaning |
|---|---|
| `cluster_id` | stable public cluster ID (`C000001`, ...); null for excluded DBSCAN noise |
| `is_noise` | whether scikit-learn DBSCAN labelled the point noise |
| `is_core` | DBSCAN core-point flag |
| `snap_distance` | straight-line distance from the original point to the network |
| `snapped_x` | x coordinate of the snapped network position |
| `snapped_y` | y coordinate of the snapped network position |

By default, noise remains in the output with `cluster_id = null`. With `noise_policy="singleton"`, each noise point gets its own cluster ID while `is_noise` remains true.

## CLI

```powershell
netdbscan run `
  --points "data\points.parquet" `
  --boundary "data\boundary.parquet" `
  --network "data\roads.parquet" `
  --point-id-col "point_id" `
  --eps 1000 `
  --min-samples 5 `
  --output "output\clustered_points.parquet"
```

GeoPackage inputs work directly. A one-layer GeoPackage needs no extra option:

```powershell
netdbscan run `
  --points "data\points.parquet" `
  --boundary "data\boundary.gpkg" `
  --network "data\roads.gpkg" `
  --eps 1000 `
  --output "output\clustered_points.parquet"
```

For a multi-layer GeoPackage, specify the layer explicitly with `--points-layer`,
`--boundary-layer`, or `--network-layer`. Shapefile inputs also work.

Use `--max-snap-distance` to refuse observations that are implausibly far from the supplied network, and `--max-neighbor-pairs` to cap sparse-neighbour memory use.

## Python API

```python
import geopandas as gpd
from netdbscan import NetDBSCANConfig, cluster_geodataframes

points = gpd.read_parquet("points.parquet")
boundary = gpd.read_parquet("boundary.parquet")
network = gpd.read_parquet("roads.parquet")

clustered = cluster_geodataframes(
    points=points,
    boundary=boundary,
    network=network,
    point_id_col="point_id",
    config=NetDBSCANConfig(eps=1000, min_samples=5),
)
```

## Distance definition

For observations `i` and `j`, let `s_i` and `s_j` be their snapped positions on the network. The DBSCAN metric is

```text
d(i, j) = shortest network distance from s_i to s_j.
```

Point-to-network snap distance is **not** added to this metric. It is reported separately as QA metadata. Points on disconnected network components can never be DBSCAN neighbours.

The neighbour search is sparse: the implementation discovers only pairs with network distance `<= eps`, then passes that sparse precomputed relation to scikit-learn's standard DBSCAN implementation. Observations at exactly the same snapped position are compressed to one weighted position for DBSCAN and expanded back to the original observations afterwards.

## Determinism

Observations are sorted by the string representation of `point_id` before clustering. Public cluster IDs are then numbered by each cluster's smallest member key. This makes public labels stable under input row permutation. As in standard DBSCAN, an ambiguous border point reachable from two clusters follows expansion order; here that order is deterministic because it is tied to `point_id` ordering.

## Prior art

Network-constrained DBSCAN is not a new clustering algorithm. Relevant precedents include Geoff Boeing's `network-clustering` example and the NS-DBSCAN literature. `netdbscan` focuses on a reusable Python implementation that keeps observations at continuous positions along network arcs, avoids a dense all-pairs road-distance matrix, and delegates DBSCAN itself to scikit-learn.

- https://github.com/gboeing/network-clustering
- Geoff Boeing (2018), *Network-Based Spatial Clustering* (blog + open GitHub example).
- Wang, Ren, Luo & Tian (2019), *NS-DBSCAN: A Density-Based Clustering Algorithm in Network Space*, ISPRS International Journal of Geo-Information 8(5):218.

## License

MIT.
