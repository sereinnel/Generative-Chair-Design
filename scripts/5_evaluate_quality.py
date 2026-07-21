"""Evaluate generated point clouds with symmetric Chamfer-L2 distance."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = (".npy", ".ply", ".pcd", ".xyz")


@dataclass(frozen=True)
class PointCloudRecord:
    """A sampled point cloud and its nearest-neighbor index."""

    name: str
    path: Path
    points: np.ndarray
    tree: cKDTree


def load_point_cloud(path: Path) -> np.ndarray:
    """Load a point cloud as a finite float32 array with shape ``(N, 3)``."""
    if path.suffix.lower() == ".npy":
        points = np.load(path).astype(np.float32, copy=False)
    else:
        point_cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(point_cloud.points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path} has shape {points.shape}; expected (N, 3)")
    if len(points) == 0:
        raise ValueError(f"{path} contains no points")
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN or infinite values")

    return points


def sample_points(
    points: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a fixed number of points, with replacement when necessary."""
    replace = len(points) < num_points
    indices = rng.choice(len(points), size=num_points, replace=replace)
    return points[indices].astype(np.float32, copy=False)


def symmetric_chamfer_l2(
    first: np.ndarray,
    second: np.ndarray,
    first_tree: cKDTree | None = None,
    second_tree: cKDTree | None = None,
) -> float:
    """Average the two directed mean nearest-neighbor Euclidean distances."""
    first_index = first_tree if first_tree is not None else cKDTree(first)
    second_index = second_tree if second_tree is not None else cKDTree(second)

    first_to_second = second_index.query(first, k=1)[0].mean()
    second_to_first = first_index.query(second, k=1)[0].mean()
    return 0.5 * float(first_to_second + second_to_first)


def find_point_cloud_files(directory: Path) -> list[Path]:
    """Return one file per stem, preferring NPY when several formats exist."""
    if not directory.is_dir():
        raise FileNotFoundError(f"directory not found: {directory}")

    by_stem: dict[str, Path] = {}
    extension_priority = {".npy": 0, ".ply": 1, ".pcd": 2, ".xyz": 3}

    for path in sorted(directory.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        current = by_stem.get(path.stem)
        if current is None or extension_priority[suffix] < extension_priority[
            current.suffix.lower()
        ]:
            by_stem[path.stem] = path

    return sorted(by_stem.values())


def load_records(
    paths: Iterable[Path],
    num_points: int,
    rng: np.random.Generator,
    label: str,
) -> list[PointCloudRecord]:
    """Load, sample and index a collection of point clouds."""
    records: list[PointCloudRecord] = []

    for path in tqdm(list(paths), desc=f"Loading {label}"):
        try:
            points = sample_points(load_point_cloud(path), num_points, rng)
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.warning("Skipping %s: %s", path.name, error)
            continue

        records.append(
            PointCloudRecord(
                name=path.name,
                path=path,
                points=points,
                tree=cKDTree(points),
            )
        )

    return records


def nearest_reference_results(
    generated: list[PointCloudRecord],
    references: list[PointCloudRecord],
) -> list[dict[str, object]]:
    """Find the closest reference cloud for every generated sample."""
    results: list[dict[str, object]] = []

    for sample in tqdm(generated, desc="Nearest-reference evaluation"):
        best_reference: PointCloudRecord | None = None
        best_distance = float("inf")

        for reference in references:
            distance = symmetric_chamfer_l2(
                sample.points,
                reference.points,
                sample.tree,
                reference.tree,
            )
            if distance < best_distance:
                best_distance = distance
                best_reference = reference

        if best_reference is None:
            raise RuntimeError("reference set is empty")

        results.append(
            {
                "generated_file": sample.name,
                "generated_path": str(sample.path),
                "nearest_reference": best_reference.name,
                "chamfer_l2": best_distance,
            }
        )

    return results


def pairwise_diversity(records: list[PointCloudRecord]) -> list[float]:
    """Compute upper-triangular pairwise Chamfer-L2 distances."""
    values: list[float] = []

    for first_index in tqdm(range(len(records)), desc="Pairwise diversity"):
        first = records[first_index]
        for second in records[first_index + 1 :]:
            values.append(
                symmetric_chamfer_l2(
                    first.points,
                    second.points,
                    first.tree,
                    second.tree,
                )
            )

    return values


def summarize(
    nearest_results: list[dict[str, object]],
    pairwise_values: list[float],
) -> dict[str, float | int | None]:
    """Aggregate evaluation metrics."""
    nearest = np.asarray(
        [result["chamfer_l2"] for result in nearest_results],
        dtype=np.float64,
    )
    pairwise = np.asarray(pairwise_values, dtype=np.float64)

    return {
        "num_generated": int(len(nearest_results)),
        "mean_nearest_reference_chamfer_l2": (
            float(nearest.mean()) if len(nearest) else None
        ),
        "median_nearest_reference_chamfer_l2": (
            float(np.median(nearest)) if len(nearest) else None
        ),
        "std_nearest_reference_chamfer_l2": (
            float(nearest.std()) if len(nearest) else None
        ),
        "min_nearest_reference_chamfer_l2": (
            float(nearest.min()) if len(nearest) else None
        ),
        "max_nearest_reference_chamfer_l2": (
            float(nearest.max()) if len(nearest) else None
        ),
        "mean_pairwise_chamfer_l2": (
            float(pairwise.mean()) if len(pairwise) else None
        ),
        "median_pairwise_chamfer_l2": (
            float(np.median(pairwise)) if len(pairwise) else None
        ),
        "std_pairwise_chamfer_l2": (
            float(pairwise.std()) if len(pairwise) else None
        ),
    }


def write_csv(path: Path, results: list[dict[str, object]]) -> None:
    """Write per-sample nearest-reference results."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "generated_file",
                "generated_path",
                "nearest_reference",
                "chamfer_l2",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **result,
                    "chamfer_l2": f"{float(result['chamfer_l2']):.8f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "generated",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "normalized_npy" / "test",
        help="Held-out split used as validation data in the original experiment.",
    )
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluation.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluation_summary.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.num_points <= 0:
        raise SystemExit("--num-points must be positive")

    generated_paths = find_point_cloud_files(args.generated_dir)
    reference_paths = find_point_cloud_files(args.reference_dir)

    if not generated_paths:
        raise SystemExit(f"No generated point clouds found in {args.generated_dir}")
    if not reference_paths:
        raise SystemExit(f"No reference point clouds found in {args.reference_dir}")

    rng = np.random.default_rng(args.seed)
    generated = load_records(
        generated_paths,
        args.num_points,
        rng,
        "generated clouds",
    )
    references = load_records(
        reference_paths,
        args.num_points,
        rng,
        "reference clouds",
    )

    if not generated or not references:
        raise SystemExit("Evaluation requires at least one valid generated and reference cloud.")

    nearest_results = nearest_reference_results(generated, references)
    pairwise_values = pairwise_diversity(generated)
    summary = summarize(nearest_results, pairwise_values)

    write_csv(args.out_csv, nearest_results)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Evaluation summary")
    for name, value in summary.items():
        if isinstance(value, float):
            LOGGER.info("%s: %.8f", name, value)
        else:
            LOGGER.info("%s: %s", name, value)

    LOGGER.info("Per-sample results: %s", args.out_csv)
    LOGGER.info("Summary: %s", args.summary_json)


if __name__ == "__main__":
    main()
