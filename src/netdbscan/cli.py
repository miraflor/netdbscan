from __future__ import annotations

from pathlib import Path

import typer

from .pipeline import NetDBSCANConfig, cluster_files

app = typer.Typer(add_completion=False, help="DBSCAN clustering by shortest-path distance along a line network.")


@app.callback()
def main() -> None:
    """netdbscan command line interface."""


@app.command()
def run(
    points: Path = typer.Option(..., "--points", help="Point layer (.parquet/.geoparquet/.gpkg/.shp)."),
    boundary: Path = typer.Option(..., "--boundary", help="Polygon boundary (.parquet/.geoparquet/.gpkg/.shp)."),
    network: Path = typer.Option(..., "--network", help="Line network (.parquet/.geoparquet/.gpkg/.shp); CRS must be projected."),
    points_layer: str | None = typer.Option(None, "--points-layer", help="Layer name when --points is a multi-layer .gpkg."),
    boundary_layer: str | None = typer.Option(None, "--boundary-layer", help="Layer name when --boundary is a multi-layer .gpkg."),
    network_layer: str | None = typer.Option(None, "--network-layer", help="Layer name when --network is a multi-layer .gpkg."),
    output: Path = typer.Option(..., "--output", help="Output grouped points (.parquet/.geoparquet)."),
    eps: float = typer.Option(..., "--eps", help="DBSCAN road-network radius in network-CRS units."),
    min_samples: int = typer.Option(5, "--min-samples"),
    point_id_col: str = typer.Option("point_id", "--point-id-col"),
    noise_policy: str = typer.Option("exclude", "--noise-policy", help="exclude or singleton."),
    max_snap_distance: float | None = typer.Option(None, "--max-snap-distance"),
    max_neighbor_pairs: int = typer.Option(10_000_000, "--max-neighbor-pairs"),
    force: bool = typer.Option(False, "--force", help="Replace an existing output file."),
) -> None:
    config = NetDBSCANConfig(
        eps=eps,
        min_samples=min_samples,
        noise_policy=noise_policy,
        max_snap_distance=max_snap_distance,
        max_neighbor_pairs=max_neighbor_pairs,
    )
    result = cluster_files(
        points_path=points,
        boundary_path=boundary,
        network_path=network,
        output_path=output,
        config=config,
        point_id_col=point_id_col,
        points_layer=points_layer,
        boundary_layer=boundary_layer,
        network_layer=network_layer,
        force=force,
    )
    n_clusters = len({x for x in result["cluster_id"].tolist() if x is not None})
    typer.echo(
        f"wrote {len(result)} boundary-covered points; "
        f"clusters={n_clusters}; noise={int(result['is_noise'].sum())}; output={output}"
    )


if __name__ == "__main__":
    app()
