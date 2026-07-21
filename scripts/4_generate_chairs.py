"""Generate chair point clouds from a trained PointNet-VAE checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import open3d as o3d
import torch

from model import PointNetVAE

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract model weights from a state dict or a training checkpoint."""
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
    """Load a trained model and switch it to evaluation mode."""
    model = PointNetVAE(
        num_points=num_points,
        latent_dim=latent_dim,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.eval()
    return model


def save_ply(points: np.ndarray, output_path: Path) -> None:
    """Save a point array as a PLY point cloud."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        points.astype(np.float64, copy=False)
    )

    if not o3d.io.write_point_cloud(str(output_path), point_cloud):
        raise OSError(f"failed to write {output_path}")


@torch.inference_mode()
def generate_samples(
    model: PointNetVAE,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
    latent_scale: float,
    output_format: str,
) -> None:
    """Decode random latent vectors and save the resulting point clouds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(num_samples):
        latent = torch.randn(
            1,
            model.latent_dim,
            device=device,
        ) * latent_scale
        points = model.decode(latent)[0].cpu().numpy().astype(np.float32)

        stem = output_dir / f"chair_{index:03d}"
        if output_format in {"npy", "both"}:
            np.save(stem.with_suffix(".npy"), points)
        if output_format in {"ply", "both"}:
            save_ply(points, stem.with_suffix(".ply"))

    LOGGER.info("Generated %d samples in %s", num_samples, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "pointnet_vae_best.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "generated",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=16_384)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument(
        "--format",
        choices=("npy", "ply", "both"),
        default="both",
        dest="output_format",
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
    if args.latent_scale <= 0:
        raise SystemExit("--latent-scale must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    model = load_model(
        args.checkpoint,
        device,
        args.num_points,
        args.latent_dim,
    )
    generate_samples(
        model,
        device,
        args.output_dir,
        args.num_samples,
        args.latent_scale,
        args.output_format,
    )


if __name__ == "__main__":
    main()
