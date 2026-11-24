#!/usr/bin/env python3
"""A minimal PaiNN-style scalar-vector interaction and mixing block.

This follows the operational structure of the PaiNN paper and SchNetPack:
invariant scalar channels q, equivariant vector channels mu, directional edge
messages, invariant contractions, residual updates, and an invariant energy head.
It is a teaching implementation, not a drop-in replacement for SchNetPack.
"""

from __future__ import annotations

import torch
from torch import nn

from lesson05_message_passing import RadialBasis, build_neighbor_graph, rotation_z


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def scatter_sum(
    edge_values: torch.Tensor,
    receivers: torch.Tensor,
    atom_count: int,
) -> torch.Tensor:
    output_shape = (atom_count, *edge_values.shape[1:])
    output = torch.zeros(output_shape, dtype=edge_values.dtype, device=edge_values.device)
    output.index_add_(0, receivers, edge_values)
    return output


class PaiNNInteraction(nn.Module):
    """Interatomic message block: neighbors update scalar and vector channels."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.features = features
        self.context = nn.Sequential(
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, 3 * features),
        )

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        filters: torch.Tensor,
        directions: torch.Tensor,
        receivers: torch.Tensor,
        senders: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_count = scalar.shape[0]

        sender_context = self.context(scalar[senders])
        edge_context = filters * sender_context
        scalar_part, direction_gate, vector_gate = torch.split(
            edge_context, self.features, dim=-1
        )

        delta_scalar = scatter_sum(scalar_part, receivers, atom_count)

        directional_message = directions[:, :, None] * direction_gate[:, None, :]
        transported_vector = vector[senders] * vector_gate[:, None, :]
        delta_vector = scatter_sum(
            directional_message + transported_vector,
            receivers,
            atom_count,
        )

        return scalar + delta_scalar, vector + delta_vector


class PaiNNMixing(nn.Module):
    """Intra-atomic block: invariant contractions couple scalar/vector channels."""

    def __init__(self, features: int, epsilon: float = 1.0e-8) -> None:
        super().__init__()
        self.features = features
        self.epsilon = epsilon
        self.vector_mix = nn.Linear(features, 2 * features, bias=False)
        self.context = nn.Sequential(
            nn.Linear(2 * features, features),
            nn.SiLU(),
            nn.Linear(features, 3 * features),
        )

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixed_vector = self.vector_mix(vector)
        vector_u, vector_v = torch.split(mixed_vector, self.features, dim=-1)

        vector_norm = torch.sqrt(
            torch.sum(vector_v.square(), dim=1) + self.epsilon
        )
        invariant_context = torch.cat((scalar, vector_norm), dim=-1)
        gates = self.context(invariant_context)
        scalar_gate, vector_gate, dot_gate = torch.split(
            gates, self.features, dim=-1
        )

        invariant_dot = torch.sum(vector_u * vector_v, dim=1)
        delta_scalar = scalar_gate + dot_gate * invariant_dot
        delta_vector = vector_gate[:, None, :] * vector_u

        return scalar + delta_scalar, vector + delta_vector


class MinimalPaiNN(nn.Module):
    def __init__(
        self,
        features: int = 24,
        radial_features: int = 12,
        cutoff: float = 3.0,
        max_atomic_number: int = 118,
    ) -> None:
        super().__init__()
        self.features = features
        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_atomic_number + 1, features)
        self.radial_basis = RadialBasis(radial_features, cutoff)
        self.filter_network = nn.Linear(radial_features, 3 * features, bias=False)
        self.interaction = PaiNNInteraction(features)
        self.mixing = PaiNNMixing(features)
        self.energy_head = nn.Sequential(
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, 1),
        )

    def representations(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        receivers, senders = build_neighbor_graph(positions, self.cutoff)
        displacement = positions[senders] - positions[receivers]
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        if torch.any(distances == 0.0).item():
            raise ValueError("distinct atoms cannot occupy identical positions")
        directions = displacement / distances[:, None]

        scalar = self.embedding(atomic_numbers)
        vector = torch.zeros(
            (atomic_numbers.shape[0], 3, self.features),
            dtype=scalar.dtype,
            device=scalar.device,
        )
        filters = self.filter_network(self.radial_basis(distances))

        scalar, vector = self.interaction(
            scalar,
            vector,
            filters,
            directions,
            receivers,
            senders,
        )
        return self.mixing(scalar, vector)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        scalar, _ = self.representations(atomic_numbers, positions)
        return self.energy_head(scalar).sum()


def energy_and_forces(
    model: nn.Module,
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    *,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = positions.detach().clone().requires_grad_(True)
    energy = model(atomic_numbers, coordinates)
    forces = -torch.autograd.grad(
        energy, coordinates, create_graph=create_graph
    )[0]
    return energy, forces


def main() -> None:
    torch.manual_seed(23)
    model = MinimalPaiNN().to(device=DEVICE, dtype=DTYPE)
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
    rotation = rotation_z(51.0)
    rotated_positions = positions @ rotation.T

    scalar, vector = model.representations(atomic_numbers, positions)
    rotated_scalar, rotated_vector = model.representations(
        atomic_numbers, rotated_positions
    )
    expected_rotated_vector = torch.einsum("ab,nbf->naf", rotation, vector)

    energy, forces = energy_and_forces(model, atomic_numbers, positions)
    rotated_energy, rotated_forces = energy_and_forces(
        model, atomic_numbers, rotated_positions
    )
    expected_rotated_forces = forces @ rotation.T

    print(f"scalar_shape={tuple(scalar.shape)}")
    print(f"vector_shape={tuple(vector.shape)}")
    print("\nEquivariance checks")
    print(f"scalar invariance error       : {torch.max(torch.abs(scalar - rotated_scalar)).item():.3e}")
    print(f"vector equivariance error     : {torch.max(torch.abs(expected_rotated_vector - rotated_vector)).item():.3e}")
    print(f"energy invariance error       : {abs(energy.item() - rotated_energy.item()):.3e}")
    print(f"force equivariance error      : {torch.max(torch.abs(expected_rotated_forces - rotated_forces)).item():.3e}")


if __name__ == "__main__":
    main()
