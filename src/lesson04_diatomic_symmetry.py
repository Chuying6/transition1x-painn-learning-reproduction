#!/usr/bin/env python3
"""Demonstrate invariant energy and equivariant forces for a 3D diatomic.

The model receives Cartesian coordinates but predicts energy only from the
interatomic distance.  This builds translation and rotation invariance into the
representation.  Forces are obtained as -grad(E, R), so they rotate with the
molecule and sum to zero for the isolated pair.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


DTYPE = torch.float64
DEVICE = torch.device("cpu")
EQUILIBRIUM_DISTANCE = 1.0
DISSOCIATION_ENERGY = 4.5
INVERSE_LENGTH = 1.8


def bond_displacement(positions: torch.Tensor) -> torch.Tensor:
    difference = positions[:, 1, :] - positions[:, 0, :]
    distance = torch.linalg.vector_norm(difference, dim=-1, keepdim=True)
    return distance - EQUILIBRIUM_DISTANCE


def reference_energy(positions: torch.Tensor) -> torch.Tensor:
    q = bond_displacement(positions)
    return DISSOCIATION_ENERGY * (1.0 - torch.exp(-INVERSE_LENGTH * q)).square()


def energy_and_forces(
    energy_function: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    positions: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = positions.detach().clone().requires_grad_(True)
    energy = energy_function(coordinates)
    gradient = torch.autograd.grad(
        energy.sum(), coordinates, create_graph=create_graph
    )[0]
    return energy, -gradient


class DistanceEnergyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        q = bond_displacement(positions)
        return self.network(q)


def make_linear_structures(count: int = 96) -> torch.Tensor:
    q = torch.linspace(-0.25, 0.60, count, dtype=DTYPE, device=DEVICE)
    distance = EQUILIBRIUM_DISTANCE + q
    positions = torch.zeros((count, 2, 3), dtype=DTYPE, device=DEVICE)
    positions[:, 0, 0] = -0.5 * distance
    positions[:, 1, 0] = 0.5 * distance
    return positions


def train(epochs: int = 2500) -> DistanceEnergyNetwork:
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")

    torch.manual_seed(11)
    model = DistanceEnergyNetwork().to(device=DEVICE, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)

    structures = make_linear_structures()
    reference_e, reference_f = energy_and_forces(
        reference_energy, structures, create_graph=False
    )
    reference_e = reference_e.detach()
    reference_f = reference_f.detach()

    for epoch in range(1, epochs + 1):
        predicted_e, predicted_f = energy_and_forces(
            model, structures, create_graph=True
        )
        energy_loss = torch.mean((predicted_e - reference_e).square())
        force_loss = torch.mean((predicted_f - reference_f).square())
        loss = energy_loss + 0.05 * force_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch in {1, 500, 1000, 1500, 2000, epochs}:
            print(
                f"epoch={epoch:4d} energy_rmse={energy_loss.sqrt().item():.6f} "
                f"force_rmse={force_loss.sqrt().item():.6f}"
            )
    return model


def rotation_z(angle_degrees: float) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )


def symmetry_checks(model: nn.Module) -> None:
    positions = torch.tensor(
        [[[-0.70, 0.0, 0.0], [0.70, 0.0, 0.0]]],
        dtype=DTYPE,
        device=DEVICE,
    )
    translation = torch.tensor([2.0, -1.0, 0.5], dtype=DTYPE, device=DEVICE)
    rotation = rotation_z(63.0)

    energy, forces = energy_and_forces(model, positions, create_graph=False)
    shifted_energy, shifted_forces = energy_and_forces(
        model, positions + translation, create_graph=False
    )
    rotated_positions = positions @ rotation.T
    rotated_energy, rotated_forces = energy_and_forces(
        model, rotated_positions, create_graph=False
    )
    expected_rotated_forces = forces @ rotation.T

    print("\nSymmetry checks")
    print(f"translation energy difference : {abs(energy.item() - shifted_energy.item()):.3e} eV")
    print(f"translation force difference  : {torch.max(torch.abs(forces - shifted_forces)).item():.3e} eV/angstrom")
    print(f"rotation energy difference    : {abs(energy.item() - rotated_energy.item()):.3e} eV")
    print(f"force equivariance error      : {torch.max(torch.abs(expected_rotated_forces - rotated_forces)).item():.3e} eV/angstrom")
    print(f"net force magnitude           : {torch.linalg.vector_norm(forces.sum(dim=1)).item():.3e} eV/angstrom")


def main() -> None:
    model = train()
    symmetry_checks(model)


if __name__ == "__main__":
    main()
