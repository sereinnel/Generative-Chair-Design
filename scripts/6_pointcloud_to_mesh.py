"""Reconstruct polygonal meshes from generated point clouds."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = (".npy", ".ply", ".pcd", ".xyz")


@dataclass(frozen=True)
class ReconstructionConfig:
    """Parameters used by the surface-reconstruction pipeline."""

    poisson_depth: int = 10
    density_quantile: float = 0.1
    target_triangles: int | None = 15_000
    outlier_neighbors: int = 50
    outlier_std_ratio: float = 1.0
    normal_neighbors: int = 30
    max_points: int | None = 200_000
    ball_pivoting_radii: tuple[float, ...] = (0.01, 0.02, 0.04)
    keep_intermediate: bool = False


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """Load a supported point-cloud file."""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported point-cloud format: {path.suffix}")

    if path.suffix.lower() == ".npy":
        points = np.load(path).astype(np.float64, copy=False)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"expected an array with shape (N, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("point cloud contains NaN or infinite values")

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
    else:
        point_cloud = o3d.io.read_point_cloud(str(path))

    if point_cloud.is_empty():
        raise ValueError("point cloud is empty")

    return point_cloud


def preprocess_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    config: ReconstructionConfig,
) -> o3d.geometry.PointCloud:
    """Downsample, remove outliers and estimate consistently oriented normals."""
    if config.max_points and len(point_cloud.points) > config.max_points:
        ratio = config.max_points / len(point_cloud.points)
        point_cloud = point_cloud.random_down_sample(ratio)

    try:
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=config.outlier_neighbors,
            std_ratio=config.outlier_std_ratio,
        )
    except RuntimeError as error:
        LOGGER.warning("Outlier removal failed: %s", error)

    if point_cloud.is_empty():
        raise ValueError("point cloud became empty during preprocessing")

    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(
            knn=config.normal_neighbors
        )
    )

    try:
        point_cloud.orient_normals_consistent_tangent_plane(
            k=config.normal_neighbors
        )
    except RuntimeError as error:
        LOGGER.warning("Consistent normal orientation failed: %s", error)

    return point_cloud


def poisson_reconstruction(
    point_cloud: o3d.geometry.PointCloud,
    depth: int,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Run Poisson surface reconstruction."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=depth,
    )
    return mesh, np.asarray(densities)


def remove_low_density_vertices(
    mesh: o3d.geometry.TriangleMesh,
    densities: np.ndarray,
    quantile: float,
) -> o3d.geometry.TriangleMesh:
    """Remove vertices below a selected Poisson-density quantile."""
    if not 0.0 <= quantile < 1.0:
        raise ValueError("density quantile must be in [0, 1)")
    if len(densities) != len(mesh.vertices):
        raise ValueError("density array does not match the number of mesh vertices")

    threshold = float(np.quantile(densities, quantile))
    mesh.remove_vertices_by_mask(densities < threshold)
    mesh.remove_unreferenced_vertices()
    return mesh


def simplify_mesh(
    mesh: o3d.geometry.TriangleMesh,
    target_triangles: int | None,
) -> o3d.geometry.TriangleMesh:
    """Reduce triangle count when a positive target is provided."""
    if target_triangles is None or len(mesh.triangles) <= target_triangles:
        return mesh

    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles
    )
    simplified.remove_unreferenced_vertices()
    return simplified


def ball_pivoting_reconstruction(
    point_cloud: o3d.geometry.PointCloud,
    radii: Sequence[float],
) -> o3d.geometry.TriangleMesh:
    """Run Ball Pivoting as a fallback when Poisson reconstruction fails."""
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud,
        o3d.utility.DoubleVector(list(radii)),
    )
    return mesh


def write_mesh(path: Path, mesh: o3d.geometry.TriangleMesh) -> None:
    """Validate and save a triangle mesh."""
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError("reconstruction produced an empty mesh")

    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(
        str(path),
        mesh,
        write_triangle_uvs=False,
    ):
        raise OSError(f"failed to write {path}")


def reconstruct_file(
    source: Path,
    output_dir: Path,
    config: ReconstructionConfig,
) -> tuple[str, Path | None, str | None]:
    """Reconstruct one point cloud, using Ball Pivoting as a fallback."""
    point_cloud = preprocess_point_cloud(load_point_cloud(source), config)

    if config.keep_intermediate:
        cleaned_path = output_dir / f"{source.stem}_cleaned.ply"
        o3d.io.write_point_cloud(str(cleaned_path), point_cloud)

    try:
        mesh, densities = poisson_reconstruction(
            point_cloud,
            config.poisson_depth,
        )
        mesh = remove_low_density_vertices(
            mesh,
            densities,
            config.density_quantile,
        )
        method = "poisson"
        output_path = output_dir / f"{source.stem}.obj"
    except (RuntimeError, ValueError) as poisson_error:
        LOGGER.warning(
            "%s: Poisson reconstruction failed (%s); trying Ball Pivoting",
            source.name,
            poisson_error,
        )
        try:
            mesh = ball_pivoting_reconstruction(
                point_cloud,
                config.ball_pivoting_radii,
            )
        except RuntimeError as ball_pivoting_error:
            return (
                source.name,
                None,
                f"Poisson: {poisson_error}; Ball Pivoting: {ball_pivoting_error}",
            )

        method = "ball-pivoting"
        output_path = output_dir / f"{source.stem}_bp.obj"

    mesh = simplify_mesh(mesh, config.target_triangles)

    try:
        write_mesh(output_path, mesh)
    except (OSError, ValueError) as error:
        return source.name, None, str(error)

    return source.name, output_path, method


def find_point_cloud_files(directory: Path) -> list[Path]:
    """Find supported files, preferring NPY when stems are duplicated."""
    if not directory.is_dir():
        raise FileNotFoundError(f"input directory not found: {directory}")

    priority = {".npy": 0, ".ply": 1, ".pcd": 2, ".xyz": 3}
    by_stem: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        current = by_stem.get(path.stem)
        if current is None or priority[suffix] < priority[current.suffix.lower()]:
            by_stem[path.stem] = path

    return sorted(by_stem.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "generated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "meshes",
    )
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--density-quantile", type=float, default=0.1)
    parser.add_argument("--target-triangles", type=int, default=15_000)
    parser.add_argument("--outlier-neighbors", type=int, default=50)
    parser.add_argument("--outlier-std-ratio", type=float, default=1.0)
    parser.add_argument("--normal-neighbors", type=int, default=30)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument(
        "--ball-pivoting-radii",
        type=float,
        nargs="+",
        default=(0.01, 0.02, 0.04),
    )
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open each reconstructed mesh in the Open3D viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = ReconstructionConfig(
        poisson_depth=args.depth,
        density_quantile=args.density_quantile,
        target_triangles=(
            args.target_triangles if args.target_triangles > 0 else None
        ),
        outlier_neighbors=args.outlier_neighbors,
        outlier_std_ratio=args.outlier_std_ratio,
        normal_neighbors=args.normal_neighbors,
        max_points=args.max_points if args.max_points > 0 else None,
        ball_pivoting_radii=tuple(args.ball_pivoting_radii),
        keep_intermediate=args.keep_intermediate,
    )

    files = find_point_cloud_files(args.input_dir)
    if not files:
        raise SystemExit(f"No supported point clouds found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    succeeded = 0

    for source in tqdm(files, desc="Reconstructing meshes"):
        name, output_path, message = reconstruct_file(
            source,
            args.output_dir,
            config,
        )

        if output_path is None:
            LOGGER.error("%s: %s", name, message)
            continue

        succeeded += 1
        LOGGER.info("%s -> %s (%s)", name, output_path.name, message)

        if args.show:
            mesh = o3d.io.read_triangle_mesh(str(output_path))
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh])

    LOGGER.info(
        "Finished: reconstructed %d of %d point clouds",
        succeeded,
        len(files),
    )


if __name__ == "__main__":
    main()
