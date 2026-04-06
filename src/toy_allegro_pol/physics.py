from __future__ import annotations

import math
from typing import Any

import torch


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_reference_positions(config: dict[str, Any]) -> torch.Tensor:
    """Centered equilateral triangle used as the reference geometry."""
    side = float(config["system"]["reference_side_length"])
    height = math.sqrt(3.0) * side / 2.0
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [side, 0.0],
            [0.5 * side, height],
        ],
        dtype=_torch_dtype(config["dtype"]),
    )
    return positions - positions.mean(dim=0, keepdim=True)


def _pair_indices(num_atoms: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx_i = []
    idx_j = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            idx_i.append(i)
            idx_j.append(j)
    return (
        torch.tensor(idx_i, dtype=torch.long, device=device),
        torch.tensor(idx_j, dtype=torch.long, device=device),
    )


def _as_batched(
    positions: torch.Tensor, electric_field: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    single_sample = positions.ndim == 2
    if single_sample:
        positions = positions.unsqueeze(0)
        electric_field = electric_field.unsqueeze(0)
    return positions, electric_field, single_sample


def _from_batched(value: torch.Tensor, single_sample: bool) -> torch.Tensor:
    if single_sample:
        return value[0]
    return value


def _center_positions(positions: torch.Tensor) -> torch.Tensor:
    return positions - positions.mean(dim=0, keepdim=True)


def _rotation_matrix_2d(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    c = torch.cos(theta).to(dtype)
    s = torch.sin(theta).to(dtype)
    return torch.stack(
        [
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ]
    )


def sample_positions(config: dict[str, Any], generator: torch.Generator) -> torch.Tensor:
    dtype = _torch_dtype(config["dtype"])
    base = build_reference_positions(config)
    noise_std = float(config["sampling"]["position_noise_std"])
    noise = torch.randn(base.shape, generator=generator, dtype=dtype) * noise_std
    theta = 2.0 * math.pi * torch.rand((), generator=generator, dtype=dtype)
    rotation = _rotation_matrix_2d(theta, dtype)
    positions = (base + noise) @ rotation.T
    return _center_positions(positions)


def sample_electric_field(
    config: dict[str, Any], generator: torch.Generator
) -> torch.Tensor:
    dtype = _torch_dtype(config["dtype"])
    field_max = float(config["sampling"]["field_max"])
    return (2.0 * torch.rand(2, generator=generator, dtype=dtype) - 1.0) * field_max


def _pair_geometry(
    positions: torch.Tensor, electric_field: torch.Tensor, config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    positions, electric_field, single_sample = _as_batched(positions, electric_field)
    _, num_atoms, _ = positions.shape
    idx_i, idx_j = _pair_indices(num_atoms, positions.device)

    rij = positions[:, idx_i] - positions[:, idx_j]
    dij = torch.linalg.norm(rij, dim=-1).clamp_min(1e-12)
    e_sq = (electric_field**2).sum(dim=-1, keepdim=True)

    return {
        "positions": positions,
        "electric_field": electric_field,
        "single_sample": torch.tensor(single_sample),
        "idx_i": idx_i,
        "idx_j": idx_j,
        "rij": rij,
        "dij": dij,
        "e_sq": e_sq,
    }


def compute_energy(positions: torch.Tensor, electric_field: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    geom = _pair_geometry(positions, electric_field, config)
    dtype = geom["positions"].dtype

    spring_k = torch.tensor(config["physics"]["spring_k"], dtype=dtype, device=geom["positions"].device)
    spring_length = torch.tensor(config["physics"]["spring_length"], dtype=dtype, device=geom["positions"].device)
    repulsion_amp = torch.tensor(config["physics"]["repulsion_amp"], dtype=dtype, device=geom["positions"].device)
    repulsion_sigma = torch.tensor(config["physics"]["repulsion_sigma"], dtype=dtype, device=geom["positions"].device)
    alpha0 = torch.tensor(config["physics"]["alpha0"], dtype=dtype, device=geom["positions"].device)
    alpha1 = torch.tensor(config["physics"]["alpha1"], dtype=dtype, device=geom["positions"].device)
    alpha_rho = torch.tensor(config["physics"]["alpha_rho"], dtype=dtype, device=geom["positions"].device)
    charges = torch.tensor(config["system"]["charges"], dtype=dtype, device=geom["positions"].device)

    spring_term = 0.5 * spring_k * (geom["dij"] - spring_length) ** 2
    repulsion_term = repulsion_amp * torch.exp(
        -(geom["dij"] ** 2) / (2.0 * repulsion_sigma**2)
    )
    alpha_pair_term = torch.exp(-(geom["dij"] ** 2) / (2.0 * alpha_rho**2))

    u_geom = (spring_term + repulsion_term).sum(dim=-1)
    dipole = (geom["positions"] * charges.view(1, -1, 1)).sum(dim=1)
    alpha = alpha0 + alpha1 * alpha_pair_term.sum(dim=-1)

    field_coupling = (geom["electric_field"] * dipole).sum(dim=-1)
    field_energy = 0.5 * alpha * geom["e_sq"].squeeze(-1)
    total_energy = u_geom - field_coupling - field_energy

    return _from_batched(total_energy, bool(geom["single_sample"].item()))


def compute_response(
    positions: torch.Tensor, electric_field: torch.Tensor, config: dict[str, Any]
) -> torch.Tensor:
    geom = _pair_geometry(positions, electric_field, config)
    dtype = geom["positions"].dtype
    charges = torch.tensor(config["system"]["charges"], dtype=dtype, device=geom["positions"].device)
    alpha0 = torch.tensor(config["physics"]["alpha0"], dtype=dtype, device=geom["positions"].device)
    alpha1 = torch.tensor(config["physics"]["alpha1"], dtype=dtype, device=geom["positions"].device)
    alpha_rho = torch.tensor(config["physics"]["alpha_rho"], dtype=dtype, device=geom["positions"].device)

    alpha_pair_term = torch.exp(-(geom["dij"] ** 2) / (2.0 * alpha_rho**2))
    alpha = alpha0 + alpha1 * alpha_pair_term.sum(dim=-1, keepdim=True)
    dipole = (geom["positions"] * charges.view(1, -1, 1)).sum(dim=1)
    response = dipole + alpha * geom["electric_field"]
    return _from_batched(response, bool(geom["single_sample"].item()))


def compute_forces(
    positions: torch.Tensor, electric_field: torch.Tensor, config: dict[str, Any]
) -> torch.Tensor:
    geom = _pair_geometry(positions, electric_field, config)
    dtype = geom["positions"].dtype
    device = geom["positions"].device
    batch_size, num_atoms, dim = geom["positions"].shape

    spring_k = torch.tensor(config["physics"]["spring_k"], dtype=dtype, device=device)
    spring_length = torch.tensor(config["physics"]["spring_length"], dtype=dtype, device=device)
    repulsion_amp = torch.tensor(config["physics"]["repulsion_amp"], dtype=dtype, device=device)
    repulsion_sigma = torch.tensor(config["physics"]["repulsion_sigma"], dtype=dtype, device=device)
    alpha1 = torch.tensor(config["physics"]["alpha1"], dtype=dtype, device=device)
    alpha_rho = torch.tensor(config["physics"]["alpha_rho"], dtype=dtype, device=device)
    charges = torch.tensor(config["system"]["charges"], dtype=dtype, device=device)

    repulsion_exp = torch.exp(-(geom["dij"] ** 2) / (2.0 * repulsion_sigma**2))
    alpha_exp = torch.exp(-(geom["dij"] ** 2) / (2.0 * alpha_rho**2))

    coeff = (
        spring_k * (geom["dij"] - spring_length) / geom["dij"]
        - (repulsion_amp / repulsion_sigma**2) * repulsion_exp
        + (alpha1 / (2.0 * alpha_rho**2)) * geom["e_sq"] * alpha_exp
    )

    pair_force = -coeff.unsqueeze(-1) * geom["rij"]
    forces = torch.zeros((batch_size, num_atoms, dim), dtype=dtype, device=device)

    for pair_idx, (i_idx, j_idx) in enumerate(zip(geom["idx_i"].tolist(), geom["idx_j"].tolist())):
        forces[:, i_idx, :] += pair_force[:, pair_idx, :]
        forces[:, j_idx, :] -= pair_force[:, pair_idx, :]

    forces = forces + charges.view(1, -1, 1) * geom["electric_field"].unsqueeze(1)
    return _from_batched(forces, bool(geom["single_sample"].item()))


def compute_labels(
    positions: torch.Tensor, electric_field: torch.Tensor, config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    return {
        "total_energy": compute_energy(positions, electric_field, config),
        "forces": compute_forces(positions, electric_field, config),
        "response": compute_response(positions, electric_field, config),
    }


def is_stable_sample(
    positions: torch.Tensor, electric_field: torch.Tensor, config: dict[str, Any]
) -> bool:
    geom = _pair_geometry(positions, electric_field, config)
    min_distance = float(config["sampling"]["min_distance"])
    max_distance = float(config["sampling"]["max_distance"])
    max_abs_energy = float(config["sampling"]["max_abs_energy"])

    dij = geom["dij"][0]
    if torch.any(dij < min_distance):
        return False
    if torch.any(dij > max_distance):
        return False

    energy = compute_energy(positions, electric_field, config)
    if abs(float(energy)) > max_abs_energy:
        return False

    return True
