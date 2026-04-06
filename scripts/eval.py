from __future__ import annotations

import argparse
import json
import os
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
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "mpl-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from toy_allegro_pol.dataset import ToyPolarDataset, dataset_path  # noqa: E402
from toy_allegro_pol.eval_utils import (  # noqa: E402
    autograd_consistency_check,
    collect_predictions,
    field_sweep,
    regression_metrics,
    save_field_sweep_plot,
    save_parity_plot,
)
from toy_allegro_pol.model import ToyGeneralizedPotential  # noqa: E402


def torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: Path, config: dict[str, Any]) -> ToyGeneralizedPotential:
    model = ToyGeneralizedPotential(config).to(dtype=torch_dtype(config["dtype"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the toy generalized potential model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a saved model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "eval",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ToyPolarDataset(ROOT / dataset_path(config, args.split), dtype=config["dtype"])
    loader = DataLoader(dataset, batch_size=int(config["train"]["batch_size"]), shuffle=False)
    model = load_model(args.checkpoint, config)

    collected = collect_predictions(model, loader)
    metrics = {
        "energy": regression_metrics(collected["pred_energy"], collected["true_energy"]),
        "forces": regression_metrics(collected["pred_forces"], collected["true_forces"]),
        "response": regression_metrics(collected["pred_response"], collected["true_response"]),
    }

    sample = dataset[0]
    fd_metrics = autograd_consistency_check(model, sample, epsilon=1.0e-6)
    metrics["finite_difference"] = fd_metrics

    save_parity_plot(
        collected["pred_energy"],
        collected["true_energy"],
        str(args.output_dir / "energy_parity.png"),
        "Energy parity",
        "True energy",
        "Predicted energy",
    )
    save_parity_plot(
        collected["pred_forces"],
        collected["true_forces"],
        str(args.output_dir / "force_parity.png"),
        "Force parity",
        "True force components",
        "Predicted force components",
    )
    save_parity_plot(
        collected["pred_response"],
        collected["true_response"],
        str(args.output_dir / "response_parity.png"),
        "Response parity",
        "True response components",
        "Predicted response components",
    )

    field_values = torch.linspace(
        -float(config["sampling"]["field_max"]),
        float(config["sampling"]["field_max"]),
        21,
        dtype=torch_dtype(config["dtype"]),
    )
    sweep = field_sweep(model, sample["positions"], sample["species"], field_values)
    save_field_sweep_plot(
        sweep,
        str(args.output_dir / "field_sweep_energy.png"),
        str(args.output_dir / "field_sweep_response.png"),
    )

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"saved plots and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
