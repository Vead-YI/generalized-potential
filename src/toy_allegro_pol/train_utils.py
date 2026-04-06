from __future__ import annotations

from typing import Any

import torch
from torch import nn


def predict_energy_force_response(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    create_graph: bool,
) -> dict[str, torch.Tensor]:
    positions = batch["positions"].clone().detach().requires_grad_(True)
    electric_field = batch["electric_field"].clone().detach().requires_grad_(True)

    model_batch = dict(batch)
    model_batch["positions"] = positions
    model_batch["electric_field"] = electric_field

    total_energy = model(model_batch)
    force = -torch.autograd.grad(
        total_energy.sum(),
        positions,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    response = -torch.autograd.grad(
        total_energy.sum(),
        electric_field,
        create_graph=create_graph,
    )[0]

    return {
        "total_energy": total_energy,
        "forces": force,
        "response": response,
    }


def compute_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    energy_loss = torch.mean((pred["total_energy"] - batch["total_energy"]) ** 2)
    force_loss = torch.mean((pred["forces"] - batch["forces"]) ** 2)
    response_loss = torch.mean((pred["response"] - batch["response"]) ** 2)

    total_loss = (
        float(config["train"]["energy_weight"]) * energy_loss
        + float(config["train"]["force_weight"]) * force_loss
        + float(config["train"]["response_weight"]) * response_loss
    )

    metrics = {
        "loss": float(total_loss.detach()),
        "energy_mse": float(energy_loss.detach()),
        "force_mse": float(force_loss.detach()),
        "response_mse": float(response_loss.detach()),
    }
    return total_loss, metrics
