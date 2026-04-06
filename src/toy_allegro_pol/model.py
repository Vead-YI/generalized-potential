from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _make_mlp(input_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)


class ToyGeneralizedPotential(nn.Module):
    """A tiny energy model with separate atom and pair channels.

    The interface mirrors the key allegro-pol idea:
    `forward(batch) -> total_energy`, while forces and response come from autograd.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        num_species = len(config["system"]["species"])
        embed_dim = int(config["model"]["species_embed_dim"])
        atom_hidden = int(config["model"]["atom_hidden_dim"])
        pair_hidden = int(config["model"]["pair_hidden_dim"])
        atom_layers = int(config["model"]["atom_layers"])
        pair_layers = int(config["model"]["pair_layers"])

        self.species_embed = nn.Embedding(num_species, embed_dim)
        self.atom_mlp = _make_mlp(
            input_dim=2 + 2 + 1 + embed_dim,
            hidden_dim=atom_hidden,
            num_layers=atom_layers,
        )
        self.pair_mlp = _make_mlp(
            input_dim=2 + 1 + 1 + 1 + 2 * embed_dim,
            hidden_dim=pair_hidden,
            num_layers=pair_layers,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        positions = batch["positions"]
        species = batch["species"]
        electric_field = batch["electric_field"]

        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
            species = species.unsqueeze(0)
            electric_field = electric_field.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size, num_atoms, _ = positions.shape
        centered_positions = positions - positions.mean(dim=1, keepdim=True)
        species_embed = self.species_embed(species)

        e_norm = torch.linalg.norm(electric_field, dim=-1, keepdim=True)
        atom_field = electric_field.unsqueeze(1).expand(-1, num_atoms, -1)
        atom_feat = torch.cat(
            [centered_positions, atom_field, e_norm.unsqueeze(1).expand(-1, num_atoms, -1), species_embed],
            dim=-1,
        )
        atom_energy = self.atom_mlp(atom_feat).squeeze(-1).sum(dim=1)

        pair_terms = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                rij = centered_positions[:, i, :] - centered_positions[:, j, :]
                dij = torch.linalg.norm(rij, dim=-1, keepdim=True)
                uij = rij / dij.clamp_min(1e-12)
                e_dot_u = (electric_field * uij).sum(dim=-1, keepdim=True)
                pair_feat = torch.cat(
                    [
                        rij,
                        dij,
                        e_dot_u,
                        e_norm,
                        species_embed[:, i, :],
                        species_embed[:, j, :],
                    ],
                    dim=-1,
                )
                pair_terms.append(self.pair_mlp(pair_feat).squeeze(-1))

        pair_energy = torch.stack(pair_terms, dim=1).sum(dim=1)
        total_energy = atom_energy + pair_energy

        if squeeze:
            return total_energy[0]
        return total_energy
