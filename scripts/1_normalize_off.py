"""Normalize ModelNet chair meshes before point-cloud sampling."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MeshExtent:
    """Axis-aligned dimensions of a normalized mesh."""

    width: float
    depth: float
    height: float


def normalize_mesh(source: Path, destination: Path) -> MeshExtent:
    """Normalize one mesh and save it as an OFF file.

    The lowest point is moved to ``z = 0``. All coordinates are then
    scaled by the original height, which preserves the aspect ratio,
    and the mesh is centered in the XY plane.
    """
    mesh = o3d.io.read_triangle_mesh(str(source))

    if mesh.is_empty() or len(mesh.vertices) == 0:
        raise ValueError("mesh contains no vertices")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"expected vertices with shape (N, 3), got {vertices.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh contains NaN or infinite coordinates")

    vertices[:, 2] -= vertices[:, 2].min()
    height = float(vertices[:, 2].max())

    if height <= 1e-8:
        raise ValueError("mesh height is zero")

    vertices /= height
    vertices[:, :2] -= vertices[:, :2].mean(axis=0)

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not o3d.io.write_triangle_mesh(str(destination), mesh):
        raise OSError(f"failed to write {destination}")

    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    return MeshExtent(
        width=float(extent[0]),
        depth=float(extent[1]),
        height=float(extent[2]),
    )


def process_split(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    """Normalize every OFF file in a dataset split."""
    if not input_dir.is_dir():
        LOGGER.warning("Skipping missing directory: %s", input_dir)
        return 0, 0

    source_files = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() == ".off"
    )
    if not source_files:
        LOGGER.warning("No OFF files found in %s", input_dir)
        return 0, 0

    succeeded = 0
    for source in tqdm(source_files, desc=f"Normalizing {input_dir.name}"):
        destination = output_dir / source.name
        try:
            extent = normalize_mesh(source, destination)
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.warning("Could not process %s: %s", source.name, error)
            continue

        LOGGER.debug(
            "%s: %.3f x %.3f x %.3f",
            source.name,
            extent.width,
            extent.depth,
            extent.height,
        )
        succeeded += 1

    return succeeded, len(source_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize ModelNet chair meshes to unit height."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing dataset split folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "normalized_off",
        help="Directory for normalized OFF files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "test"),
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file dimensions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    total_succeeded = 0
    total_files = 0

    for split in args.splits:
        succeeded, count = process_split(
            args.input_root / split,
            args.output_root / split,
        )
        total_succeeded += succeeded
        total_files += count
        LOGGER.info("%s: processed %d of %d files", split, succeeded, count)

    if total_files == 0:
        raise SystemExit("No input meshes were found.")

    LOGGER.info("Finished: processed %d of %d files", total_succeeded, total_files)


if __name__ == "__main__":
    main()
