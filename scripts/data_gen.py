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

from toy_allegro_pol.physics import (  # noqa: E402
    compute_labels,
    is_stable_sample,
    sample_electric_field,
    sample_positions,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _split_seed(base_seed: int, offset: int) -> int:
    return int(base_seed + 1009 * offset)


def generate_split(
    split_name: str,
    split_size: int,
    config: dict[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    generator = torch.Generator().manual_seed(seed)
    max_attempts = int(config["sampling"]["max_attempts_per_sample"])
    species_ids = np.asarray(config["system"]["species_ids"], dtype=np.int64)

    positions = []
    electric_fields = []
    total_energies = []
    forces = []
    responses = []

    num_rejected = 0
    for sample_idx in range(split_size):
        accepted = False
        for _ in range(max_attempts):
            pos = sample_positions(config, generator)
            efield = sample_electric_field(config, generator)
            if not is_stable_sample(pos, efield, config):
                num_rejected += 1
                continue

            labels = compute_labels(pos, efield, config)
            positions.append(pos.detach().cpu().numpy())
            electric_fields.append(efield.detach().cpu().numpy())
            total_energies.append(float(labels["total_energy"]))
            forces.append(labels["forces"].detach().cpu().numpy())
            responses.append(labels["response"].detach().cpu().numpy())
            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                f"Failed to generate a stable {split_name} sample after "
                f"{max_attempts} attempts at sample index {sample_idx}."
            )

    return {
        "positions": np.asarray(positions),
        "species": np.broadcast_to(species_ids, (split_size, species_ids.shape[0])).copy(),
        "electric_field": np.asarray(electric_fields),
        "total_energy": np.asarray(total_energies),
        "forces": np.asarray(forces),
        "response": np.asarray(responses),
        "num_rejected": np.asarray([num_rejected], dtype=np.int64),
    }


def save_split(
    split_name: str,
    split_data: dict[str, np.ndarray],
    config: dict[str, Any],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{config['dataset']['output_name']}_{split_name}.npz"
    metadata = {
        "split": split_name,
        "dtype": config["dtype"],
        "species": config["system"]["species"],
        "charges": config["system"]["charges"],
    }
    np.savez_compressed(out_path, **split_data, metadata=json.dumps(metadata))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a toy allegro-pol style dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="Path to the YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    base_seed = int(config["seed"])
    output_dir = ROOT / config["dataset"]["output_dir"]

    split_sizes = {
        "train": int(config["dataset"]["train_size"]),
        "val": int(config["dataset"]["val_size"]),
        "test": int(config["dataset"]["test_size"]),
    }

    for split_offset, (split_name, split_size) in enumerate(split_sizes.items()):
        split_data = generate_split(
            split_name=split_name,
            split_size=split_size,
            config=config,
            seed=_split_seed(base_seed, split_offset),
        )
        out_path = save_split(split_name, split_data, config, output_dir)
        print(
            f"[{split_name}] saved {split_size} samples to {out_path} "
            f"(rejected={int(split_data['num_rejected'][0])})"
        )


if __name__ == "__main__":
    main()
