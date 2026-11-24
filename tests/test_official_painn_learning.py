from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/transition1x-matplotlib-test-cache")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from official_painn_learning import (  # noqa: E402
    OfficialPaiNNEnergy,
    complete_directed_edges,
    make_inputs,
    predict_energy_and_forces,
    reference_energy,
    reference_labels,
    sample_structures,
)
from schnetpack import properties  # noqa: E402


class OfficialPaiNNLearningTests(unittest.TestCase):
    def test_structure_sampling_is_reproducible(self) -> None:
        first = sample_structures(count=4, seed=10)
        second = sample_structures(count=4, seed=10)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertEqual(first.shape, (4, 3, 3))

    def test_complete_edges_do_not_cross_molecules(self) -> None:
        idx_i, idx_j = complete_directed_edges(batch_size=2)
        edges = set(zip(idx_i.tolist(), idx_j.tolist(), strict=True))
        expected = {
            (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1),
            (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4),
        }
        self.assertEqual(edges, expected)

    def test_graph_inputs_have_consistent_shapes_and_directions(self) -> None:
        positions = sample_structures(count=2, seed=10)
        inputs = make_inputs(positions)
        self.assertEqual(inputs[properties.Z].shape, (6,))
        self.assertEqual(inputs[properties.R].shape, (6, 3))
        self.assertEqual(inputs[properties.Rij].shape, (12, 3))
        self.assertEqual(inputs[properties.idx_m].tolist(), [0, 0, 0, 1, 1, 1])
        first_i = inputs[properties.idx_i][0]
        first_j = inputs[properties.idx_j][0]
        expected = inputs[properties.R][first_j] - inputs[properties.R][first_i]
        torch.testing.assert_close(inputs[properties.Rij][0], expected)

    def test_reference_forces_match_finite_difference(self) -> None:
        positions = sample_structures(count=1, seed=10)
        _, forces = reference_labels(positions)
        step = 1.0e-4
        plus = positions.clone()
        minus = positions.clone()
        plus[0, 1, 0] += step
        minus[0, 1, 0] -= step
        finite_difference = -(reference_energy(plus) - reference_energy(minus)) / (2.0 * step)
        torch.testing.assert_close(
            forces[0, 1, 0], finite_difference[0], rtol=2.0e-3, atol=2.0e-4
        )

    def test_untrained_model_has_rigid_motion_symmetry(self) -> None:
        torch.manual_seed(7)
        model = OfficialPaiNNEnergy()
        positions = sample_structures(count=2, seed=20)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        translation = torch.tensor([1.2, -0.7, 0.4])
        transformed = torch.einsum("ij,baj->bai", rotation, positions) + translation
        energy, forces = predict_energy_and_forces(model, positions, create_graph=False)
        moved_energy, moved_forces = predict_energy_and_forces(
            model, transformed, create_graph=False
        )
        expected_forces = torch.einsum("ij,baj->bai", rotation, forces)
        torch.testing.assert_close(moved_energy, energy, rtol=0.0, atol=1.0e-6)
        torch.testing.assert_close(
            moved_forces, expected_forces, rtol=0.0, atol=1.0e-6
        )


if __name__ == "__main__":
    unittest.main()
