"""Train the PointNet-style VAE on normalized chair point clouds."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from model import PointNetVAE

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for the original experiment."""

    data_root: Path = PROJECT_ROOT / "data" / "normalized_npy"
    train_subdir: str = "train"
    validation_subdir: str = "test"
    model_dir: Path = PROJECT_ROOT / "models"

    output_points: int = 16_384
    input_points: int = 4096
    loss_points: int = 4096
    latent_dim: int = 256

    batch_size: int = 8
    epochs: int = 23
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    beta: float = 1e-4
    gradient_clip: float = 1.0

    num_workers: int = 4
    print_every: int = 20
    validation_batches: int = 50
    seed: int = 42
    use_amp: bool = True


class PointCloudDataset(Dataset[torch.Tensor]):
    """Load NPY point clouds and sample a fixed number of input points."""

    def __init__(self, directory: Path, num_points: int) -> None:
        if not directory.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {directory}")

        self.paths = sorted(directory.glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"no NPY files found in {directory}")

        self.num_points = num_points

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        points = np.load(self.paths[index]).astype(np.float32, copy=False)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"{self.paths[index]} has shape {points.shape}; expected (N, 3)"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"{self.paths[index]} contains non-finite values")

        replace = points.shape[0] < self.num_points
        indices = np.random.choice(
            points.shape[0],
            size=self.num_points,
            replace=replace,
        )
        return torch.from_numpy(points[indices])


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic NumPy seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sample_point_dimension(points: torch.Tensor, count: int) -> torch.Tensor:
    """Select the same random point indices for every item in a batch."""
    num_available = points.shape[1]

    if num_available >= count:
        indices = torch.randperm(num_available, device=points.device)[:count]
    else:
        indices = torch.randint(
            num_available,
            size=(count,),
            device=points.device,
        )

    return points[:, indices, :]


def bidirectional_chamfer_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the sum of the two directed mean Chamfer distances.

    This matches the reconstruction objective used in the original run.
    Distances are Euclidean rather than squared Euclidean distances.
    """
    distances = torch.cdist(predicted, target)
    predicted_to_target = distances.amin(dim=2).mean()
    target_to_predicted = distances.amin(dim=1).mean()
    return predicted_to_target + target_to_predicted


def kl_divergence(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    """Return the batch-mean KL divergence from the unit Gaussian."""
    per_sample = -0.5 * torch.sum(
        1 + log_variance - mean.square() - log_variance.exp(),
        dim=1,
    )
    return per_sample.mean()


def train_epoch(
    model: PointNetVAE,
    dataloader: DataLoader[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean total loss."""
    model.train()
    total_loss = 0.0
    total_items = 0
    amp_enabled = config.use_amp and device.type == "cuda"

    for step, batch in enumerate(dataloader, start=1):
        target = batch.to(device, non_blocking=True)
        encoder_input = target.transpose(1, 2).contiguous()

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            reconstruction, mean, log_variance = model(encoder_input)
            reconstruction_subset = sample_point_dimension(
                reconstruction,
                config.loss_points,
            )
            target_subset = sample_point_dimension(target, config.loss_points)

            reconstruction_loss = bidirectional_chamfer_loss(
                reconstruction_subset,
                target_subset,
            )
            latent_loss = kl_divergence(mean, log_variance)
            loss = reconstruction_loss + config.beta * latent_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = target.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        if step % config.print_every == 0:
            LOGGER.info(
                "step %d/%d, mean loss %.6f",
                step,
                len(dataloader),
                total_loss / total_items,
            )

    return total_loss / total_items


def symmetric_chamfer_l2(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Average the two directed mean nearest-neighbor distances."""
    first_tree = cKDTree(first)
    second_tree = cKDTree(second)
    first_to_second = second_tree.query(first, k=1)[0].mean()
    second_to_first = first_tree.query(second, k=1)[0].mean()
    return 0.5 * float(first_to_second + second_to_first)


@torch.inference_mode()
def evaluate_reconstruction(
    model: PointNetVAE,
    dataloader: DataLoader[torch.Tensor],
    config: TrainingConfig,
    device: torch.device,
) -> tuple[float, float]:
    """Measure reconstruction Chamfer-L2 on the validation split."""
    model.eval()
    values: list[float] = []

    for batch_index, batch in enumerate(dataloader):
        if batch_index >= config.validation_batches:
            break

        target = batch.to(device, non_blocking=True)
        encoder_input = target.transpose(1, 2).contiguous()
        reconstruction, _, _ = model(encoder_input)

        reconstruction_subset = sample_point_dimension(
            reconstruction,
            config.loss_points,
        ).cpu().numpy()
        target_subset = sample_point_dimension(
            target,
            config.loss_points,
        ).cpu().numpy()

        for predicted, reference in zip(reconstruction_subset, target_subset):
            values.append(symmetric_chamfer_l2(predicted, reference))

    if not values:
        raise RuntimeError("validation produced no measurements")

    return float(np.mean(values)), float(np.std(values))


def checkpoint_payload(
    epoch: int,
    model: PointNetVAE,
    optimizer: torch.optim.Optimizer,
    validation_chamfer: float,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Build a restartable training checkpoint."""
    serializable_config = asdict(config)
    serializable_config["data_root"] = str(config.data_root)
    serializable_config["model_dir"] = str(config.model_dir)

    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "validation_chamfer_l2": validation_chamfer,
        "config": serializable_config,
    }


def train(config: TrainingConfig) -> None:
    """Train the model and save the best and latest checkpoints."""
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    train_dataset = PointCloudDataset(
        config.data_root / config.train_subdir,
        config.input_points,
    )
    validation_dataset = PointCloudDataset(
        config.data_root / config.validation_subdir,
        config.input_points,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": config.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    model = PointNetVAE(
        num_points=config.output_points,
        latent_dim=config.latent_dim,
    ).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Model parameters: %.1f million", parameter_count / 1e6)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")

    config.model_dir.mkdir(parents=True, exist_ok=True)
    (config.model_dir / "training_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "data_root": str(config.data_root),
                "model_dir": str(config.model_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_chamfer = float("inf")
    best_path = config.model_dir / "pointnet_vae_best.pth"
    latest_path = config.model_dir / "pointnet_vae_latest.pth"

    for epoch in range(1, config.epochs + 1):
        started_at = time.perf_counter()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            config,
            device,
        )
        validation_mean, validation_std = evaluate_reconstruction(
            model,
            validation_loader,
            config,
            device,
        )

        elapsed = time.perf_counter() - started_at
        LOGGER.info(
            "epoch %d/%d | train loss %.6f | validation Chamfer-L2 "
            "%.6f +/- %.6f | %.1f s",
            epoch,
            config.epochs,
            train_loss,
            validation_mean,
            validation_std,
            elapsed,
        )

        payload = checkpoint_payload(
            epoch,
            model,
            optimizer,
            validation_mean,
            config,
        )
        torch.save(payload, latest_path)

        if validation_mean < best_chamfer:
            best_chamfer = validation_mean
            torch.save(model.state_dict(), best_path)
            LOGGER.info(
                "Saved new best model at epoch %d (Chamfer-L2 %.6f)",
                epoch,
                best_chamfer,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--input-points", type=int)
    parser.add_argument("--loss-points", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig()

    overrides = {
        "data_root": args.data_root,
        "model_dir": args.model_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "input_points": args.input_points,
        "loss_points": args.loss_points,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }

    for field, value in overrides.items():
        if value is not None:
            setattr(config, field, value)

    if args.disable_amp:
        config.use_amp = False

    positive_fields = (
        "epochs",
        "batch_size",
        "latent_dim",
        "input_points",
        "loss_points",
    )
    for field in positive_fields:
        if getattr(config, field) <= 0:
            raise ValueError(f"{field} must be positive")

    return config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
