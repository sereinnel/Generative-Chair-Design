"""Visualize random generations or deterministic VAE reconstructions."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from model import PointNetVAE

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract model weights from common checkpoint formats."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")

    for key in ("model_state", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint

    raise ValueError("checkpoint does not contain a recognizable model state")


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    num_points: int,
    latent_dim: int,
) -> PointNetVAE:
    """Load a trained model for visualization."""
    model = PointNetVAE(
        num_points=num_points,
        latent_dim=latent_dim,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.eval()
    return model


def sample_input(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a fixed-size encoder input from an NPY point cloud."""
    replace = len(points) < count
    indices = rng.choice(len(points), size=count, replace=replace)
    return points[indices].astype(np.float32, copy=False)


@torch.inference_mode()
def random_generations(
    model: PointNetVAE,
    device: torch.device,
    count: int,
    latent_scale: float,
) -> list[np.ndarray]:
    """Decode random samples from the unit Gaussian prior."""
    latent = torch.randn(
        count,
        model.latent_dim,
        device=device,
    ) * latent_scale
    return [
        points.astype(np.float32)
        for points in model.decode(latent).cpu().numpy()
    ]


@torch.inference_mode()
def reconstructions(
    model: PointNetVAE,
    device: torch.device,
    directory: Path,
    count: int,
    input_points: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Decode latent means for the first point clouds in a directory."""
    paths = sorted(directory.glob("*.npy"))[:count]
    if not paths:
        raise FileNotFoundError(f"No NPY files found in {directory}")

    outputs: list[np.ndarray] = []

    for path in paths:
        points = np.load(path).astype(np.float32, copy=False)
        sampled = sample_input(points, input_points, rng)
        tensor = torch.from_numpy(sampled).unsqueeze(0).to(device)
        encoder_input = tensor.transpose(1, 2).contiguous()

        mean, _ = model.encode(encoder_input)
        reconstruction = model.decode(mean)[0].cpu().numpy().astype(np.float32)
        outputs.append(reconstruction)

    return outputs


def save_ply(points: np.ndarray, path: Path) -> None:
    """Save one point cloud as PLY."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        points.astype(np.float64, copy=False)
    )
    if not o3d.io.write_point_cloud(str(path), point_cloud):
        raise OSError(f"failed to write {path}")


def set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    """Set equal axis lengths around a point cloud."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * float((maximum - minimum).max())

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_grid(
    point_clouds: list[np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    """Save point clouds in a compact Matplotlib grid."""
    columns = min(3, len(point_clouds))
    rows = math.ceil(len(point_clouds) / columns)
    figure = plt.figure(figsize=(5 * columns, 4.5 * rows))
    figure.suptitle(title)

    for index, points in enumerate(point_clouds, start=1):
        axis = figure.add_subplot(rows, columns, index, projection="3d")
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],
            s=0.4,
            alpha=0.7,
        )
        axis.set_title(f"Sample {index}")
        axis.set_axis_off()
        set_equal_axes(axis, points)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "pointnet_vae_best.pth",
    )
    parser.add_argument(
        "--mode",
        choices=("random", "reconstruct"),
        default="random",
    )
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--num-points", type=int, default=16_384)
    parser.add_argument("--input-points", type=int, default=4096)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "normalized_npy" / "test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "visualization",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    model = load_model(
        args.checkpoint,
        device,
        args.num_points,
        args.latent_dim,
    )

    if args.mode == "random":
        point_clouds = random_generations(
            model,
            device,
            args.num_samples,
            args.latent_scale,
        )
        title = "Random VAE generations"
    else:
        point_clouds = reconstructions(
            model,
            device,
            args.input_dir,
            args.num_samples,
            args.input_points,
            rng,
        )
        title = "VAE reconstructions"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, points in enumerate(point_clouds):
        stem = args.output_dir / f"{args.mode}_{index:03d}"
        np.save(stem.with_suffix(".npy"), points)
        save_ply(points, stem.with_suffix(".ply"))

    grid_path = args.output_dir / f"{args.mode}_grid.png"
    save_grid(point_clouds, grid_path, title)
    LOGGER.info("Saved %d point clouds and %s", len(point_clouds), grid_path)


if __name__ == "__main__":
    main()
