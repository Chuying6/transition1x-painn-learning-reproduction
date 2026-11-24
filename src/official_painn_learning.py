#!/usr/bin/env python3
"""Run a Learning Test with SchNetPack's official PaiNN on a known potential.

This is a learning test rather than a scientific benchmark.  The reference
energy is analytic, so we know exactly what the model is trying to learn:

    E = bond terms + angle term

Forces are generated from the reference energy with automatic differentiation.
The trained model is then checked on held-out geometries and on a rotated and
translated copy of one geometry.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

import schnetpack as spk
from schnetpack import properties


@dataclass(frozen=True)
class ReferencePotential:
    """Parameters for a water-like toy potential."""

    equilibrium_bond: float = 0.96
    equilibrium_angle: float = math.radians(104.5)
    bond_strength: float = 12.0
    angle_strength: float = 2.0


REFERENCE = ReferencePotential()


def reference_energy(positions: torch.Tensor) -> torch.Tensor:
    """Return one analytic reference energy for each three-atom structure.

    The atom order is O, H, H and ``positions`` has shape ``(batch, 3, 3)``.
    Energy units are arbitrary but used consistently throughout this test.
    """

    oxygen = positions[:, 0]
    oh_1 = positions[:, 1] - oxygen
    oh_2 = positions[:, 2] - oxygen
    r_1 = torch.linalg.vector_norm(oh_1, dim=1)
    r_2 = torch.linalg.vector_norm(oh_2, dim=1)

    cosine = (oh_1 * oh_2).sum(dim=1) / (r_1 * r_2)
    angle = torch.acos(cosine.clamp(-1.0 + 1.0e-9, 1.0 - 1.0e-9))

    bond_energy = 0.5 * REFERENCE.bond_strength * (
        (r_1 - REFERENCE.equilibrium_bond) ** 2
        + (r_2 - REFERENCE.equilibrium_bond) ** 2
    )
    angle_energy = 0.5 * REFERENCE.angle_strength * (
        angle - REFERENCE.equilibrium_angle
    ) ** 2
    return bond_energy + angle_energy


def sample_structures(count: int, seed: int) -> torch.Tensor:
    """Sample non-equilibrium geometries and randomly orient/translate them."""

    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    generator = torch.Generator().manual_seed(seed)
    r_1 = 0.76 + 0.40 * torch.rand(count, generator=generator)
    r_2 = 0.76 + 0.40 * torch.rand(count, generator=generator)
    angle = math.radians(80.0) + math.radians(50.0) * torch.rand(
        count, generator=generator
    )

    structures = torch.zeros((count, 3, 3))
    structures[:, 1, 0] = r_1
    structures[:, 2, 0] = r_2 * torch.cos(angle)
    structures[:, 2, 1] = r_2 * torch.sin(angle)

    # A QR decomposition turns random matrices into orthogonal rotations.
    random_matrices = torch.randn((count, 3, 3), generator=generator)
    rotations, _ = torch.linalg.qr(random_matrices)
    determinants = torch.linalg.det(rotations)
    rotations[:, :, 2] *= torch.where(determinants < 0.0, -1.0, 1.0)[:, None]
    structures = torch.einsum("bij,baj->bai", rotations, structures)

    translations = 2.0 * torch.rand((count, 1, 3), generator=generator) - 1.0
    return structures + translations


def reference_labels(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate reference energies and conservative forces."""

    differentiable_positions = positions.detach().clone().requires_grad_(True)
    energies = reference_energy(differentiable_positions)
    forces = -torch.autograd.grad(energies.sum(), differentiable_positions)[0]
    return energies.detach(), forces.detach()


def complete_directed_edges(
    batch_size: int, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all six directed edges for every three-atom structure."""

    if batch_size < 1:
        raise ValueError(f"batch size must be positive, got {batch_size}")
    local_i = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
    local_j = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long, device=device)
    offsets = 3 * torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
    return (local_i + offsets).reshape(-1), (local_j + offsets).reshape(-1)


def make_inputs(positions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert ``(batch, atoms, xyz)`` positions into PaiNN's graph inputs."""

    batch_size = positions.shape[0]
    flat_positions = positions.reshape(-1, 3)
    idx_i, idx_j = complete_directed_edges(batch_size, positions.device)
    molecule_index = torch.arange(
        batch_size, dtype=torch.long, device=positions.device
    ).repeat_interleave(3)

    return {
        properties.Z: torch.tensor(
            [8, 1, 1], dtype=torch.long, device=positions.device
        ).repeat(batch_size),
        properties.R: flat_positions,
        properties.Rij: flat_positions[idx_j] - flat_positions[idx_i],
        properties.idx_i: idx_i,
        properties.idx_j: idx_j,
        properties.idx_m: molecule_index,
    }


class OfficialPaiNNEnergy(nn.Module):
    """SchNetPack PaiNN representation plus a small atomwise energy head."""

    def __init__(self) -> None:
        super().__init__()
        cutoff = 3.0
        radial_basis = spk.nn.GaussianRBF(n_rbf=12, cutoff=cutoff)
        cutoff_function = spk.nn.CosineCutoff(cutoff)
        self.painn = spk.representation.PaiNN(
            n_atom_basis=24,
            n_interactions=2,
            radial_basis=radial_basis,
            cutoff_fn=cutoff_function,
        )
        self.atomwise_energy = nn.Sequential(
            nn.Linear(24, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        inputs = make_inputs(positions)
        representation = self.painn(inputs)["scalar_representation"]
        atomic_energies = self.atomwise_energy(representation).squeeze(-1)

        molecular_energies = torch.zeros(
            positions.shape[0], dtype=positions.dtype, device=positions.device
        )
        molecular_energies.index_add_(0, inputs[properties.idx_m], atomic_energies)
        return molecular_energies


def predict_energy_and_forces(
    model: nn.Module, positions: torch.Tensor, create_graph: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict energy and obtain forces as the negative coordinate gradient."""

    differentiable_positions = positions.detach().clone().requires_grad_(True)
    energies = model(differentiable_positions)
    forces = -torch.autograd.grad(
        energies.sum(), differentiable_positions, create_graph=create_graph
    )[0]
    return energies, forces


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((prediction - target) ** 2)).item()


def train_model(epochs: int, report_every: int) -> OfficialPaiNNEnergy:
    """Fit energy and force labels for a small synthetic training set."""

    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if report_every < 1:
        raise ValueError(f"report interval must be positive, got {report_every}")
    torch.manual_seed(7)
    train_positions = sample_structures(count=64, seed=10)
    train_energies, train_forces = reference_labels(train_positions)

    model = OfficialPaiNNEnergy()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        predicted_energies, predicted_forces = predict_energy_and_forces(
            model, train_positions, create_graph=True
        )
        energy_loss = torch.mean((predicted_energies - train_energies) ** 2)
        force_loss = torch.mean((predicted_forces - train_forces) ** 2)
        loss = energy_loss + 0.05 * force_loss
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % report_every == 0 or epoch == epochs:
            print(
                f"epoch {epoch:4d} | "
                f"energy RMSE {math.sqrt(energy_loss.item()):.6f} | "
                f"force RMSE {math.sqrt(force_loss.item()):.6f}"
            )

    return model


def rotation_translation_check(model: nn.Module) -> tuple[float, float]:
    """Measure energy invariance and force equivariance after a rigid motion."""

    positions = sample_structures(count=1, seed=999)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([1.2, -0.7, 0.4])
    transformed = torch.einsum("ij,baj->bai", rotation, positions) + translation

    energy, force = predict_energy_and_forces(model, positions, create_graph=False)
    moved_energy, moved_force = predict_energy_and_forces(
        model, transformed, create_graph=False
    )
    expected_force = torch.einsum("ij,baj->bai", rotation, force)
    return (
        torch.max(torch.abs(moved_energy - energy)).item(),
        torch.max(torch.abs(moved_force - expected_force)).item(),
    )


def evaluate(model: nn.Module) -> None:
    test_positions = sample_structures(count=32, seed=20)
    test_energies, test_forces = reference_labels(test_positions)
    predicted_energies, predicted_forces = predict_energy_and_forces(
        model, test_positions, create_graph=False
    )
    energy_error = rmse(predicted_energies.detach(), test_energies)
    force_error = rmse(predicted_forces.detach(), test_forces)
    invariant_error, equivariant_error = rotation_translation_check(model)

    print("\nHeld-out evaluation")
    print(f"energy RMSE                    : {energy_error:.6f}")
    print(f"force RMSE                     : {force_error:.6f}")
    print(f"rigid-motion energy error      : {invariant_error:.3e}")
    print(f"rotated-force equivariance err : {equivariant_error:.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--report-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("SchNetPack official PaiNN Learning Test")
    print("Reference target: two bond terms plus one angle term")
    print("Training target : energy + conservative force\n")
    model = train_model(epochs=args.epochs, report_every=args.report_every)
    evaluate(model)


if __name__ == "__main__":
    main()
