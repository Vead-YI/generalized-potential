from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to load configs/default.yaml") from exc


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toy_allegro_pol.dataset import ToyPolarDataset, dataset_path  # noqa: E402
from toy_allegro_pol.model import ToyGeneralizedPotential  # noqa: E402
from toy_allegro_pol.train_utils import compute_loss, predict_energy_force_response  # noqa: E402


def torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    ds = ToyPolarDataset(
        npz_path=ROOT / dataset_path(config, split),
        dtype=config["dtype"],
    )
    return DataLoader(
        ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "energy_mse": 0.0, "force_mse": 0.0, "response_mse": 0.0}
    num_batches = 0

    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)

        pred = predict_energy_force_response(
            model=model,
            batch=batch,
            create_graph=training,
        )
        loss, metrics = compute_loss(pred, batch, config)

        if training:
            loss.backward()
            optimizer.step()

        for key, value in metrics.items():
            totals[key] += value
        num_batches += 1

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the minimal toy generalized potential model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the number of epochs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_loader = make_loader(config, config["train"]["train_split"], shuffle=True)
    val_loader = make_loader(config, config["train"]["val_split"], shuffle=False)

    model = ToyGeneralizedPotential(config).to(dtype=torch_dtype(config["dtype"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    checkpoint_dir = ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"

    num_epochs = args.epochs if args.epochs is not None else int(config["train"]["epochs"])
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, config)
        val_metrics = run_epoch(model, val_loader, None, config)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, last_checkpoint_path)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, best_checkpoint_path)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_E={val_metrics['energy_mse']:.6f} "
            f"val_F={val_metrics['force_mse']:.6f} "
            f"val_P={val_metrics['response_mse']:.6f}"
        )

    print(f"saved best checkpoint to {best_checkpoint_path}")
    print(f"saved last checkpoint to {last_checkpoint_path}")


if __name__ == "__main__":
    main()
