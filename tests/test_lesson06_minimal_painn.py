from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lesson06_minimal_painn import (  # noqa: E402
    DTYPE,
    MinimalPaiNN,
    energy_and_forces,
)


def axis_angle_rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.tolist()
    skew = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=DTYPE
    )
    identity = torch.eye(3, dtype=DTYPE)
    outer = axis[:, None] * axis[None, :]
    return math.cos(angle) * identity + (1.0 - math.cos(angle)) * outer + math.sin(angle) * skew


class MinimalPaiNNTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.model = MinimalPaiNN(
            features=8, radial_features=6, cutoff=3.0
        ).to(dtype=DTYPE)
        self.atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
        self.positions = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000],
                [0.9572, 0.0000, 0.0000],
                [-0.2390, 0.9270, 0.0000],
            ],
            dtype=DTYPE,
        )

    def test_scalar_and_vector_representation_shapes(self) -> None:
        scalar, vector = self.model.representations(
            self.atomic_numbers, self.positions
        )
        self.assertEqual(scalar.shape, (3, 8))
        self.assertEqual(vector.shape, (3, 3, 8))

    def test_generic_rotation_equivariance(self) -> None:
        rotation = axis_angle_rotation(torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE), 0.73)
        rotated_positions = self.positions @ rotation.T

        scalar, vector = self.model.representations(
            self.atomic_numbers, self.positions
        )
        rotated_scalar, rotated_vector = self.model.representations(
            self.atomic_numbers, rotated_positions
        )
        expected_vector = torch.einsum("ab,nbf->naf", rotation, vector)
        torch.testing.assert_close(rotated_scalar, scalar, rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(
            rotated_vector, expected_vector, rtol=0.0, atol=1.0e-12
        )

        energy, forces = energy_and_forces(
            self.model, self.atomic_numbers, self.positions
        )
        rotated_energy, rotated_forces = energy_and_forces(
            self.model, self.atomic_numbers, rotated_positions
        )
        torch.testing.assert_close(rotated_energy, energy, rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(
            rotated_forces, forces @ rotation.T, rtol=0.0, atol=1.0e-12
        )

    def test_translation_permutation_and_net_force(self) -> None:
        energy, forces = energy_and_forces(
            self.model, self.atomic_numbers, self.positions
        )
        translation = torch.tensor([1.7, -0.4, 2.2], dtype=DTYPE)
        shifted_energy, shifted_forces = energy_and_forces(
            self.model, self.atomic_numbers, self.positions + translation
        )
        torch.testing.assert_close(shifted_energy, energy, rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(shifted_forces, forces, rtol=0.0, atol=1.0e-12)

        permutation = torch.tensor([1, 0, 2], dtype=torch.long)
        permuted_energy, permuted_forces = energy_and_forces(
            self.model,
            self.atomic_numbers[permutation],
            self.positions[permutation],
        )
        torch.testing.assert_close(permuted_energy, energy, rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(
            permuted_forces[torch.argsort(permutation)],
            forces,
            rtol=0.0,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            forces.sum(dim=0), torch.zeros(3, dtype=DTYPE), rtol=0.0, atol=1.0e-12
        )

    def test_force_loss_can_backpropagate_to_model(self) -> None:
        _, forces = energy_and_forces(
            self.model,
            self.atomic_numbers,
            self.positions,
            create_graph=True,
        )
        forces.square().mean().backward()
        nonzero_gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None and torch.count_nonzero(parameter.grad).item()
        ]
        self.assertTrue(nonzero_gradients)

    def test_coincident_atoms_raise_clear_error(self) -> None:
        positions = self.positions.clone()
        positions[1] = positions[0]
        with self.assertRaisesRegex(ValueError, "identical positions"):
            self.model(self.atomic_numbers, positions)

    def test_embedding_supports_periodic_table_atomic_numbers(self) -> None:
        atomic_numbers = torch.tensor([1, 10, 118], dtype=torch.long)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
            dtype=DTYPE,
        )
        self.assertTrue(math.isfinite(self.model(atomic_numbers, positions).item()))


if __name__ == "__main__":
    unittest.main()
