from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


class ToyPolarDataset(Dataset):
    def __init__(self, npz_path: str | Path, dtype: str = "float64"):
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        tensor_dtype = _torch_dtype(dtype)

        self.positions = torch.from_numpy(data["positions"]).to(tensor_dtype)
        self.species = torch.from_numpy(data["species"]).long()
        self.electric_field = torch.from_numpy(data["electric_field"]).to(tensor_dtype)
        self.total_energy = torch.from_numpy(data["total_energy"]).to(tensor_dtype)
        self.forces = torch.from_numpy(data["forces"]).to(tensor_dtype)
        self.response = torch.from_numpy(data["response"]).to(tensor_dtype)
        self.metadata = data.get("metadata", None)

    def __len__(self) -> int:
        return self.positions.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "positions": self.positions[idx],
            "species": self.species[idx],
            "electric_field": self.electric_field[idx],
            "total_energy": self.total_energy[idx],
            "forces": self.forces[idx],
            "response": self.response[idx],
        }


def dataset_path(config: dict[str, Any], split: str) -> Path:
    root = Path(config["dataset"]["output_dir"])
    stem = config["dataset"]["output_name"]
    return root / f"{stem}_{split}.npz"
