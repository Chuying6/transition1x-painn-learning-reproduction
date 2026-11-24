from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lesson04_diatomic_symmetry import (  # noqa: E402
    DTYPE,
    DistanceEnergyNetwork,
    energy_and_forces,
    reference_energy,
    rotation_z,
)


class DiatomicSymmetryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = DistanceEnergyNetwork().to(dtype=DTYPE)
        self.positions = torch.tensor(
            [[[0.20, -0.30, 0.50], [1.40, 0.40, -0.10]]], dtype=DTYPE
        )

    def test_translation_invariance_and_force_consistency(self) -> None:
        translation = torch.tensor([2.0, -1.0, 0.5], dtype=DTYPE)
        energy, forces = energy_and_forces(
            self.model, self.positions, create_graph=False
        )
        shifted_energy, shifted_forces = energy_and_forces(
            self.model, self.positions + translation, create_graph=False
        )
        torch.testing.assert_close(shifted_energy, energy, rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(shifted_forces, forces, rtol=0.0, atol=1.0e-12)

    def test_rotation_invariance_and_force_equivariance(self) -> None:
        rotation = rotation_z(63.0)
        energy, forces = energy_and_forces(
            self.model, self.positions, create_graph=False
        )
        rotated_energy, rotated_forces = energy_and_forces(
            self.model, self.positions @ rotation.T, create_graph=False
        )
        torch.testing.assert_close(rotated_energy, energy, rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(
            rotated_forces, forces @ rotation.T, rtol=0.0, atol=1.0e-12
        )

    def test_internal_forces_sum_to_zero(self) -> None:
        _, forces = energy_and_forces(
            self.model, self.positions, create_graph=False
        )
        torch.testing.assert_close(
            forces.sum(dim=1), torch.zeros((1, 3), dtype=DTYPE), rtol=0.0, atol=1.0e-12
        )

    def test_identical_atom_exchange_permutes_forces(self) -> None:
        energy, forces = energy_and_forces(
            self.model, self.positions, create_graph=False
        )
        exchanged_energy, exchanged_forces = energy_and_forces(
            self.model, self.positions.flip(dims=(1,)), create_graph=False
        )
        torch.testing.assert_close(exchanged_energy, energy, rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(
            exchanged_forces, forces.flip(dims=(1,)), rtol=0.0, atol=1.0e-12
        )

    def test_reference_equilibrium_has_zero_energy_and_force(self) -> None:
        equilibrium = torch.tensor(
            [[[-0.50, 0.0, 0.0], [0.50, 0.0, 0.0]]], dtype=DTYPE
        )
        energy, forces = energy_and_forces(
            reference_energy, equilibrium, create_graph=False
        )
        torch.testing.assert_close(energy, torch.zeros_like(energy), rtol=0.0, atol=1.0e-12)
        torch.testing.assert_close(forces, torch.zeros_like(forces), rtol=0.0, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
