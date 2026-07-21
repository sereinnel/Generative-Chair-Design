"""Generate chair point clouds from a trained PointNet-VAE checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from model_improved import PointNetVAE


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    num_points: int,
    latent_dim: int,
) -> PointNetVAE:
    """Load a PointNet-VAE from a state-dict or training checkpoint."""
    model = PointNetVAE(
        num_points=num_points,
        latent_dim=latent_dim,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


def save_point_cloud(points: np.ndarray, output_path: Path) -> None:
    """Save an array with shape (N, 3) as an Open3D PLY file."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected point array with shape (N, 3), got {points.shape}"
        )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        points.astype(np.float64)
    )

    success = o3d.io.write_point_cloud(
        str(output_path),
        point_cloud,
    )

    if not success:
        raise RuntimeError(f"Failed to save {output_path}")


def generate_samples(
    model: PointNetVAE,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
    latent_scale: float,
) -> None:
    """Decode random latent vectors and save the generated point clouds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index in range(num_samples):
            latent = (
                torch.randn(
                    1,
                    model.latent_dim,
                    device=device,
                )
                * latent_scale
            )

            points = model.decode(latent)[0].cpu().numpy()

            ply_path = output_dir / f"chair_{index:02d}.ply"
            npy_path = output_dir / f"chair_{index:02d}.npy"

            save_point_cloud(points, ply_path)
            np.save(npy_path, points.astype(np.float32))

            print(f"Saved: {ply_path}")
            print(f"Saved: {npy_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chair point clouds with a trained PointNet-VAE."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/model_improved_best.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generated"),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=16384,
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--latent-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        num_points=args.num_points,
        latent_dim=args.latent_dim,
    )

    generate_samples(
        model=model,
        device=device,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        latent_scale=args.latent_scale,
    )


if __name__ == "__main__":
    main()
