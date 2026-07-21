"""Sample fixed-size point clouds from normalized chair meshes."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sample_mesh(source: Path, destination: Path, num_points: int) -> None:
    """Sample one mesh surface and store the result as a float32 array."""
    mesh = o3d.io.read_triangle_mesh(str(source))

    if mesh.is_empty() or len(mesh.vertices) == 0:
        raise ValueError("mesh contains no vertices")
    if len(mesh.triangles) == 0:
        raise ValueError("mesh contains no triangles")

    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(point_cloud.points, dtype=np.float32)

    if points.shape != (num_points, 3):
        raise ValueError(
            f"expected sampled points with shape ({num_points}, 3), got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("sampled point cloud contains NaN or infinite values")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, points)


def _sample_worker(task: tuple[Path, Path, int]) -> tuple[str, str | None]:
    """Process-pool entry point."""
    source, destination, num_points = task
    try:
        sample_mesh(source, destination, num_points)
    except (OSError, RuntimeError, ValueError) as error:
        return source.name, str(error)
    return source.name, None


def convert_split(
    input_dir: Path,
    output_dir: Path,
    num_points: int,
    workers: int,
) -> tuple[int, int]:
    """Convert every normalized OFF file in one dataset split."""
    if not input_dir.is_dir():
        LOGGER.warning("Skipping missing directory: %s", input_dir)
        return 0, 0

    source_files = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() == ".off"
    )
    if not source_files:
        LOGGER.warning("No OFF files found in %s", input_dir)
        return 0, 0

    tasks = [
        (source, output_dir / f"{source.stem}.npy", num_points)
        for source in source_files
    ]
    succeeded = 0

    if workers <= 1:
        for task in tqdm(tasks, desc=f"Sampling {input_dir.name}"):
            filename, error = _sample_worker(task)
            if error is None:
                succeeded += 1
            else:
                LOGGER.warning("Could not process %s: %s", filename, error)
        return succeeded, len(tasks)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_sample_worker, task) for task in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Sampling {input_dir.name}",
        ):
            filename, error = future.result()
            if error is None:
                succeeded += 1
            else:
                LOGGER.warning("Could not process %s: %s", filename, error)

    return succeeded, len(tasks)


def validate_arrays(
    directory: Path,
    num_points: int,
    num_samples: int,
    seed: int,
) -> None:
    """Check a small deterministic sample of generated arrays."""
    paths = sorted(directory.glob("*.npy"))
    if not paths:
        LOGGER.warning("No point clouds found in %s", directory)
        return

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(
        len(paths),
        size=min(num_samples, len(paths)),
        replace=False,
    )

    for index in selected_indices:
        path = paths[int(index)]
        points = np.load(path)

        if points.shape != (num_points, 3):
            raise ValueError(f"{path} has unexpected shape {points.shape}")
        if points.dtype != np.float32:
            raise ValueError(f"{path} has unexpected dtype {points.dtype}")
        if not np.isfinite(points).all():
            raise ValueError(f"{path} contains NaN or infinite values")

        height = float(points[:, 2].max() - points[:, 2].min())
        if not np.isclose(height, 1.0, atol=0.05):
            LOGGER.warning("%s has height %.3f instead of approximately 1.0", path, height)


def write_dataset_info(
    path: Path,
    train_count: int,
    validation_count: int,
    num_points: int,
) -> None:
    """Write a concise description of the processed dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "CHAIR DATASET",
                "=============",
                f"Training samples: {train_count}",
                f"Validation samples: {validation_count}",
                f"Points per cloud: {num_points}",
                "Normalization: unit height, floor at z = 0, centered in XY",
                f"Array format: float32, shape ({num_points}, 3)",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample point clouds from normalized chair meshes."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "normalized_off",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "normalized_npy",
    )
    parser.add_argument("--num-points", type=int, default=16_384)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, multiprocessing.cpu_count())),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-samples", type=int, default=5)
    parser.add_argument(
        "--train-split",
        default="train",
        help="Name of the training split directory.",
    )
    parser.add_argument(
        "--validation-split",
        default="test",
        help="Directory used as validation data in the original experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.num_points <= 0:
        raise SystemExit("--num-points must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    train_success, train_total = convert_split(
        args.input_root / args.train_split,
        args.output_root / args.train_split,
        args.num_points,
        args.workers,
    )
    validation_success, validation_total = convert_split(
        args.input_root / args.validation_split,
        args.output_root / args.validation_split,
        args.num_points,
        args.workers,
    )

    LOGGER.info(
        "Training split: processed %d of %d files",
        train_success,
        train_total,
    )
    LOGGER.info(
        "Validation split: processed %d of %d files",
        validation_success,
        validation_total,
    )

    if train_success == 0:
        raise SystemExit("No training point clouds were created.")

    validate_arrays(
        args.output_root / args.train_split,
        args.num_points,
        args.validation_samples,
        args.seed,
    )
    write_dataset_info(
        PROJECT_ROOT / "data" / "dataset_info.txt",
        train_success,
        validation_success,
        args.num_points,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
