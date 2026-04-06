from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from toy_allegro_pol.train_utils import predict_energy_force_response  # noqa: E402


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


def batch_from_json(json_path: Path, dtype_name: str) -> dict[str, torch.Tensor]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    dtype = torch_dtype(dtype_name)
    return {
        "positions": torch.tensor(payload["positions"], dtype=dtype).unsqueeze(0),
        "species": torch.tensor(payload["species"], dtype=torch.long).unsqueeze(0),
        "electric_field": torch.tensor(payload["electric_field"], dtype=dtype).unsqueeze(0),
    }


def batch_from_dataset(config: dict[str, Any], split: str, index: int) -> dict[str, torch.Tensor]:
    dataset = ToyPolarDataset(ROOT / dataset_path(config, split), dtype=config["dtype"])
    sample = dataset[index]
    return {
        "positions": sample["positions"].unsqueeze(0),
        "species": sample["species"].unsqueeze(0),
        "electric_field": sample["electric_field"].unsqueeze(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-sample inference for the toy generalized potential model.")
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
        "--input-json",
        type=Path,
        default=None,
        help="Optional JSON file with positions, species, and electric_field.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to read from when --input-json is omitted.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index within the split when --input-json is omitted.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_model(args.checkpoint, config)

    if args.input_json is not None:
        batch = batch_from_json(args.input_json, config["dtype"])
    else:
        batch = batch_from_dataset(config, args.split, args.index)

    pred = predict_energy_force_response(model, batch, create_graph=False)
    result = {
        "total_energy": float(pred["total_energy"][0].detach()),
        "forces": pred["forces"][0].detach().cpu().numpy().tolist(),
        "response": pred["response"][0].detach().cpu().numpy().tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
