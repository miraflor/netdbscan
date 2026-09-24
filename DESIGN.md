# netdbscan — design

## Core invariant

```text
points
  → boundary filter
  → snap to supplied network
  → sparse road-distance neighbourhoods
  → scikit-learn DBSCAN
  → grouped points
```


## Input formats

GeoParquet is the canonical/recommended interchange format. GeoPackage and
Shapefile are accepted as input formats. A one-layer GeoPackage is read
automatically; when a GeoPackage contains multiple layers, the caller must
select the layer explicitly rather than relying on implicit layer order.

## Boundary

The boundary filters eligible observations. Points on the polygon boundary are retained. The line network is not clipped by the polygon.

## CRS

The network CRS is the analysis CRS and must be projected. Points and boundary may use any valid CRS and are reprojected to the network CRS. Every distance parameter is expressed in the network CRS linear units.

## Road topology

`netdbscan` accepts the topology represented by the supplied line data. It does not infer uncertain connections or silently repair topology. `MultiLineString` inputs are exploded before network construction to avoid false links between separate parts.

PySAL `spaghetti` builds the network and snaps observations. Exact endpoints are canonicalized to network vertices; interior positions on different arcs are not merged merely because their XY coordinates coincide.

## DBSCAN metric

If observation `i` snaps to network position `s_i`, then

```text
d_R(i,j) = shortest-path distance along the network from s_i to s_j.
```

Snap distance from the original observation to `s_i` is QA metadata, not part of `d_R`.

The implementation computes only pairs satisfying

```text
d_R(i,j) <= eps
```

and stores them in a sparse precomputed distance matrix. Disconnected pairs are absent. scikit-learn performs DBSCAN; this package does not reimplement DBSCAN.

## Repeated snapped positions

Several observations at one exact network position have identical distances to every other observation. They are represented once during the network search. DBSCAN receives the position with `sample_weight` equal to its multiplicity, which preserves `min_samples` semantics without materializing O(k²) zero-distance pairs.

## Determinism

`point_id` is required, unique, non-null and unique after string conversion. Points are clustered in canonical stringified-ID order. Cluster IDs are assigned as `C000001`, `C000002`, ... in order of each cluster's smallest member key.

## Noise

Default `noise_policy="exclude"` does not remove DBSCAN noise from the output. Noise receives `cluster_id = null` and `is_noise = true`.

With `noise_policy="singleton"`, each noise observation gets its own one-point cluster while retaining `is_noise = true`.

## Output

One GeoParquet point layer containing only boundary-covered input observations, preserving original attributes and adding:

```text
cluster_id
is_noise
is_core
snap_distance
snapped_x
snapped_y
```

The primary geometry remains the original observation location, reprojected to the network CRS.

## Non-goals

`netdbscan` does not:

- construct a road network from OpenStreetMap or another external service;
- clip the network to the boundary;
- add point-to-network access distance to DBSCAN distance;
- calculate centroids, Voronoi regions, service areas or downstream partitions;
- infer domain meaning from observations;
- implement alternate clustering methods;
- repair uncertain network topology.
