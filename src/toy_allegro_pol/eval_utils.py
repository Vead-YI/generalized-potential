from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_utils import predict_energy_force_response


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
) -> dict[str, np.ndarray]:
    model.eval()
    pred_energy = []
    pred_forces = []
    pred_response = []
    true_energy = []
    true_forces = []
    true_response = []

    for batch in loader:
        pred = predict_energy_force_response(model, batch, create_graph=False)
        pred_energy.append(_to_numpy(pred["total_energy"]))
        pred_forces.append(_to_numpy(pred["forces"]))
        pred_response.append(_to_numpy(pred["response"]))
        true_energy.append(_to_numpy(batch["total_energy"]))
        true_forces.append(_to_numpy(batch["forces"]))
        true_response.append(_to_numpy(batch["response"]))

    return {
        "pred_energy": np.concatenate(pred_energy, axis=0),
        "pred_forces": np.concatenate(pred_forces, axis=0),
        "pred_response": np.concatenate(pred_response, axis=0),
        "true_energy": np.concatenate(true_energy, axis=0),
        "true_forces": np.concatenate(true_forces, axis=0),
        "true_response": np.concatenate(true_response, axis=0),
    }


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    diff = pred_flat - target_flat
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    target_mean = float(np.mean(target_flat))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((target_flat - target_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def finite_difference_force(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    epsilon: float,
) -> torch.Tensor:
    positions = batch["positions"].clone().detach()
    approx = torch.zeros_like(positions)

    for atom_idx in range(positions.shape[0]):
        for dim_idx in range(positions.shape[1]):
            plus_batch = dict(batch)
            minus_batch = dict(batch)
            plus_pos = positions.clone()
            minus_pos = positions.clone()
            plus_pos[atom_idx, dim_idx] += epsilon
            minus_pos[atom_idx, dim_idx] -= epsilon
            plus_batch["positions"] = plus_pos
            minus_batch["positions"] = minus_pos
            e_plus = model(plus_batch)
            e_minus = model(minus_batch)
            approx[atom_idx, dim_idx] = -(e_plus - e_minus) / (2.0 * epsilon)

    return approx


def finite_difference_response(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    epsilon: float,
) -> torch.Tensor:
    electric_field = batch["electric_field"].clone().detach()
    approx = torch.zeros_like(electric_field)

    for dim_idx in range(electric_field.shape[0]):
        plus_batch = dict(batch)
        minus_batch = dict(batch)
        plus_e = electric_field.clone()
        minus_e = electric_field.clone()
        plus_e[dim_idx] += epsilon
        minus_e[dim_idx] -= epsilon
        plus_batch["electric_field"] = plus_e
        minus_batch["electric_field"] = minus_e
        e_plus = model(plus_batch)
        e_minus = model(minus_batch)
        approx[dim_idx] = -(e_plus - e_minus) / (2.0 * epsilon)

    return approx


def autograd_consistency_check(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    epsilon: float = 1.0e-6,
) -> dict[str, float]:
    pred = predict_energy_force_response(
        model,
        {
            key: value.unsqueeze(0) if value.ndim > 0 else value.unsqueeze(0)
            for key, value in batch.items()
        },
        create_graph=False,
    )
    force_auto = pred["forces"][0]
    response_auto = pred["response"][0]

    force_fd = finite_difference_force(model, batch, epsilon)
    response_fd = finite_difference_response(model, batch, epsilon)

    return {
        "force_fd_max_abs_err": float((force_auto - force_fd).abs().max().detach()),
        "response_fd_max_abs_err": float(
            (response_auto - response_fd).abs().max().detach()
        ),
    }


def field_sweep(
    model: torch.nn.Module,
    positions: torch.Tensor,
    species: torch.Tensor,
    field_values: torch.Tensor,
) -> dict[str, np.ndarray]:
    energies = []
    responses = []

    for field_x in field_values:
        batch = {
            "positions": positions.unsqueeze(0),
            "species": species.unsqueeze(0),
            "electric_field": torch.tensor([[field_x.item(), 0.0]], dtype=positions.dtype),
        }
        pred = predict_energy_force_response(model, batch, create_graph=False)
        energies.append(pred["total_energy"][0].item())
        responses.append(_to_numpy(pred["response"][0]))

    return {
        "field_values": _to_numpy(field_values),
        "energies": np.asarray(energies),
        "responses": np.asarray(responses),
    }


def save_parity_plot(
    pred: np.ndarray,
    target: np.ndarray,
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    lo = min(float(pred_flat.min()), float(target_flat.min()))
    hi = max(float(pred_flat.max()), float(target_flat.max()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(target_flat, pred_flat, s=8, alpha=0.5)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_field_sweep_plot(
    sweep: dict[str, np.ndarray],
    energy_path: str,
    response_path: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(sweep["field_values"], sweep["energies"], marker="o")
    ax.set_xlabel("E_x")
    ax.set_ylabel("Predicted energy")
    ax.set_title("Energy vs field sweep")
    fig.tight_layout()
    fig.savefig(energy_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(sweep["field_values"], sweep["responses"][:, 0], marker="o", label="P_x")
    ax.plot(sweep["field_values"], sweep["responses"][:, 1], marker="o", label="P_y")
    ax.set_xlabel("E_x")
    ax.set_ylabel("Predicted response")
    ax.set_title("Response vs field sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(response_path, dpi=180)
    plt.close(fig)
