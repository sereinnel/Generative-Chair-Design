"""Visualize raw meshes, normalized meshes and sampled point clouds."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    """Set equal scale on all three axes around the data center."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * float((maximum - minimum).max())

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_mesh(ax: plt.Axes, path: Path, title: str) -> None:
    """Draw a triangle mesh on a Matplotlib 3D axis."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError(f"{path} is not a valid triangle mesh")

    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=triangles,
        linewidth=0.0,
        antialiased=True,
        alpha=0.9,
    )
    ax.set_title(f"{title}\n{len(vertices)} vertices")
    set_equal_axes(ax, vertices)


def plot_point_cloud(ax: plt.Axes, path: Path) -> None:
    """Draw an NPY point cloud on a Matplotlib 3D axis."""
    points = np.load(path)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path} has shape {points.shape}; expected (N, 3)")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=points[:, 2],
        s=0.5,
        alpha=0.7,
    )
    ax.set_title(f"Sampled point cloud\n{len(points)} points")
    set_equal_axes(ax, points)


def visualize_sample(
    filename: str,
    raw_dir: Path,
    normalized_dir: Path,
    point_cloud_dir: Path,
    output_path: Path | None,
    show: bool,
) -> None:
    """Create a three-panel preprocessing comparison."""
    raw_path = raw_dir / filename
    normalized_path = normalized_dir / filename
    point_cloud_path = point_cloud_dir / f"{Path(filename).stem}.npy"

    for path in (raw_path, normalized_path, point_cloud_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    figure = plt.figure(figsize=(15, 5))
    figure.suptitle(filename)

    raw_axis = figure.add_subplot(1, 3, 1, projection="3d")
    normalized_axis = figure.add_subplot(1, 3, 2, projection="3d")
    point_axis = figure.add_subplot(1, 3, 3, projection="3d")

    plot_mesh(raw_axis, raw_path, "Raw mesh")
    plot_mesh(normalized_axis, normalized_path, "Normalized mesh")
    plot_point_cloud(point_axis, point_cloud_path)

    for axis in (raw_axis, normalized_axis, point_axis):
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")

    figure.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        LOGGER.info("Saved figure: %s", output_path)

    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filename",
        help="OFF filename to display. A random file is selected when omitted.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "preprocessing_sample.png",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_dir = PROJECT_ROOT / "data" / "raw" / args.split
    normalized_dir = PROJECT_ROOT / "data" / "normalized_off" / args.split
    point_cloud_dir = PROJECT_ROOT / "data" / "normalized_npy" / args.split

    if args.filename is None:
        candidates = sorted(raw_dir.glob("*.off"))
        if not candidates:
            raise SystemExit(f"No OFF files found in {raw_dir}")

        rng = np.random.default_rng(args.seed)
        filename = candidates[int(rng.integers(len(candidates)))].name
    else:
        filename = args.filename

    visualize_sample(
        filename,
        raw_dir,
        normalized_dir,
        point_cloud_dir,
        args.output,
        args.show,
    )


if __name__ == "__main__":
    main()
