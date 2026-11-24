from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lesson05_message_passing import (  # noqa: E402
    DTYPE,
    RadialBasis,
    ScalarMessagePassingPotential,
    build_neighbor_graph,
    cosine_cutoff,
    energy_and_forces,
    rotation_z,
)


class MessagePassingTests(unittest.TestCase):
    def test_neighbor_graph_is_directed_and_respects_cutoff(self) -> None:
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=DTYPE,
        )
        receivers, senders = build_neighbor_graph(positions, cutoff=1.5)
        edges = set(zip(receivers.tolist(), senders.tolist(), strict=True))
        self.assertEqual(edges, {(0, 1), (1, 0)})

    def test_cosine_cutoff_reaches_zero_smoothly(self) -> None:
        cutoff = 2.0
        distances = torch.tensor([0.0, 1.0, 2.0, 2.5], dtype=DTYPE)
        expected = torch.tensor([1.0, 0.5, 0.0, 0.0], dtype=DTYPE)
        torch.testing.assert_close(
            cosine_cutoff(distances, cutoff), expected, rtol=0.0, atol=1.0e-12
        )

    def test_radial_basis_requires_multiple_centers(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            RadialBasis(count=1, cutoff=3.0)

    def test_energy_and_forces_obey_geometric_symmetries(self) -> None:
        torch.manual_seed(19)
        model = ScalarMessagePassingPotential(
            hidden_features=8, radial_features=6, cutoff=3.0
        ).to(dtype=DTYPE)
        atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
        positions = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000],
                [0.9572, 0.0000, 0.0000],
                [-0.2390, 0.9270, 0.0000],
            ],
            dtype=DTYPE,
        )
        energy, forces = energy_and_forces(model, atomic_numbers, positions)

        translation = torch.tensor([2.1, -0.8, 1.4], dtype=DTYPE)
        translated_energy, translated_forces = energy_and_forces(
            model, atomic_numbers, positions + translation
        )
        torch.testing.assert_close(translated_energy, energy, rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(translated_forces, forces, rtol=0.0, atol=1.0e-12)

        rotation = rotation_z(47.0)
        rotated_energy, rotated_forces = energy_and_forces(
            model, atomic_numbers, positions @ rotation.T
        )
        torch.testing.assert_close(rotated_energy, energy, rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(
            rotated_forces, forces @ rotation.T, rtol=0.0, atol=1.0e-12
        )

        permutation = torch.tensor([1, 0, 2], dtype=torch.long)
        permuted_energy, permuted_forces = energy_and_forces(
            model, atomic_numbers[permutation], positions[permutation]
        )
        torch.testing.assert_close(permuted_energy, energy, rtol=0.0, atol=1.0e-12)
        inverse_permutation = torch.argsort(permutation)
        torch.testing.assert_close(
            permuted_forces[inverse_permutation], forces, rtol=0.0, atol=1.0e-12
        )
        torch.testing.assert_close(
            forces.sum(dim=0), torch.zeros(3, dtype=DTYPE), rtol=0.0, atol=1.0e-12
        )

    def test_embedding_supports_periodic_table_atomic_numbers(self) -> None:
        model = ScalarMessagePassingPotential(
            hidden_features=4, radial_features=4, cutoff=2.0
        ).to(dtype=DTYPE)
        atomic_numbers = torch.tensor([1, 10, 118], dtype=torch.long)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0]],
            dtype=DTYPE,
        )
        energy = model(atomic_numbers, positions)
        self.assertTrue(math.isfinite(energy.item()))


if __name__ == "__main__":
    unittest.main()
