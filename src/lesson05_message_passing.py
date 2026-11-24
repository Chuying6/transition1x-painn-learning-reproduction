#!/usr/bin/env python3
"""A minimal distance-based message-passing interatomic potential.

This is intentionally smaller than PaiNN.  It demonstrates the shared skeleton:
atomic embeddings, a cutoff graph, distance features, neighbor aggregation,
atomic energy contributions, total energy, and forces from -grad(E, R).
"""

from __future__ import annotations

import math

import torch
from torch import nn


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def build_neighbor_graph(
    positions: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return directed receiver/sender indices for pairs inside the cutoff."""
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    atom_count = positions.shape[0]
    receivers: list[int] = []
    senders: list[int] = []
    for receiver in range(atom_count):
        for sender in range(atom_count):
            if receiver == sender:
                continue
            distance = torch.linalg.vector_norm(
                positions[sender] - positions[receiver]
            )
            if distance.item() < cutoff:
                receivers.append(receiver)
                senders.append(sender)
    return (
        torch.tensor(receivers, dtype=torch.long, device=positions.device),
        torch.tensor(senders, dtype=torch.long, device=positions.device),
    )


def cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    values = 0.5 * (torch.cos(math.pi * distances / cutoff) + 1.0)
    return torch.where(distances < cutoff, values, torch.zeros_like(values))


class RadialBasis(nn.Module):
    def __init__(self, count: int, cutoff: float) -> None:
        super().__init__()
        if count < 2:
            raise ValueError(f"count must be at least 2, got {count}")
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        centers = torch.linspace(0.0, cutoff, count, dtype=DTYPE)
        self.register_buffer("centers", centers)
        spacing = cutoff / (count - 1)
        self.gamma = 1.0 / spacing**2
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        expanded = torch.exp(
            -self.gamma * (distances[:, None] - self.centers[None, :]).square()
        )
        return expanded * cosine_cutoff(distances, self.cutoff)[:, None]


class ScalarMessagePassingPotential(nn.Module):
    def __init__(
        self,
        hidden_features: int = 32,
        radial_features: int = 12,
        cutoff: float = 3.0,
        max_atomic_number: int = 118,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_atomic_number + 1, hidden_features)
        self.radial_basis = RadialBasis(radial_features, cutoff)
        self.message = nn.Sequential(
            nn.Linear(hidden_features + radial_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )
        self.atomic_energy = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        receivers, senders = build_neighbor_graph(positions, self.cutoff)
        features = self.embedding(atomic_numbers)

        displacement = positions[senders] - positions[receivers]
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        radial = self.radial_basis(distances)

        edge_input = torch.cat((features[senders], radial), dim=-1)
        messages = self.message(edge_input)

        aggregated = torch.zeros_like(features)
        aggregated.index_add_(0, receivers, messages)
        features = features + self.update(torch.cat((features, aggregated), dim=-1))

        per_atom_energy = self.atomic_energy(features)
        return per_atom_energy.sum()


def energy_and_forces(
    model: nn.Module,
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = positions.detach().clone().requires_grad_(True)
    energy = model(atomic_numbers, coordinates)
    forces = -torch.autograd.grad(energy, coordinates)[0]
    return energy, forces


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


def main() -> None:
    torch.manual_seed(19)
    model = ScalarMessagePassingPotential().to(device=DEVICE, dtype=DTYPE)

    # A bent H-O-H geometry. Atom order is O, H, H.
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long, device=DEVICE)
    positions = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )

    energy, forces = energy_and_forces(model, atomic_numbers, positions)

    translation = torch.tensor([2.1, -0.8, 1.4], dtype=DTYPE, device=DEVICE)
    translated_energy, translated_forces = energy_and_forces(
        model, atomic_numbers, positions + translation
    )

    rotation = rotation_z(47.0)
    rotated_positions = positions @ rotation.T
    rotated_energy, rotated_forces = energy_and_forces(
        model, atomic_numbers, rotated_positions
    )
    expected_rotated_forces = forces @ rotation.T

    permutation = torch.tensor([1, 0, 2], dtype=torch.long, device=DEVICE)
    permuted_energy, permuted_forces = energy_and_forces(
        model,
        atomic_numbers[permutation],
        positions[permutation],
    )

    print(f"atoms={atomic_numbers.tolist()}")
    print(f"directed_edges={len(build_neighbor_graph(positions, model.cutoff)[0])}")
    print(f"energy={energy.item():.8f}")
    print("\nSymmetry checks")
    print(f"translation energy difference : {abs(energy.item() - translated_energy.item()):.3e}")
    print(f"translation force difference  : {torch.max(torch.abs(forces - translated_forces)).item():.3e}")
    print(f"rotation energy difference    : {abs(energy.item() - rotated_energy.item()):.3e}")
    print(f"force equivariance error      : {torch.max(torch.abs(expected_rotated_forces - rotated_forces)).item():.3e}")
    print(f"permutation energy difference : {abs(energy.item() - permuted_energy.item()):.3e}")
    print(f"net force magnitude           : {torch.linalg.vector_norm(forces.sum(dim=0)).item():.3e}")

    # Restore the permuted force rows to the original atom order before comparing.
    inverse_permutation = torch.argsort(permutation)
    restored_forces = permuted_forces[inverse_permutation]
    print(f"permutation force error       : {torch.max(torch.abs(forces - restored_forces)).item():.3e}")


if __name__ == "__main__":
    main()
