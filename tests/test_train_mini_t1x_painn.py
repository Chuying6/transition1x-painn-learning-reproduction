from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/transition1x-matplotlib-test-cache")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_mini_t1x_painn import (  # noqa: E402
    Configuration,
    collate_configurations,
    composition,
    fit_element_baselines,
    full_neighbor_list,
    reaction_relative_rmse,
)


def configuration(atomic_numbers: list[int], energy: float, reaction_id: str) -> Configuration:
    atom_count = len(atomic_numbers)
    return Configuration(
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        positions=torch.zeros((atom_count, 3), dtype=torch.float32),
        energy=torch.tensor(energy, dtype=torch.float64),
        forces=torch.zeros((atom_count, 3), dtype=torch.float32),
        reaction_id=reaction_id,
    )


class TrainingUtilitiesTests(unittest.TestCase):
    def test_unknown_elements_are_not_silently_ignored(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported atomic numbers"):
            composition(torch.tensor([1, 17], dtype=torch.long))

    def test_element_baselines_recover_exact_composition_model(self) -> None:
        expected = np.asarray([-10.0, -20.0, -30.0, -40.0])
        configurations = [
            configuration([1, 1], 2 * expected[0], "H2/rxn"),
            configuration([6], expected[1], "C/rxn"),
            configuration([7], expected[2], "N/rxn"),
            configuration([8], expected[3], "O/rxn"),
        ]
        dataset = SimpleNamespace(configurations=configurations)
        baselines, condition_number = fit_element_baselines(dataset)  # type: ignore[arg-type]
        np.testing.assert_allclose(baselines, expected, rtol=0.0, atol=1.0e-12)
        self.assertGreaterEqual(condition_number, 1.0)

    def test_neighbor_list_has_no_cross_molecule_edges(self) -> None:
        molecules = [
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        ]
        idx_i, idx_j = full_neighbor_list(molecules, cutoff=1.5)
        edges = set(zip(idx_i.tolist(), idx_j.tolist(), strict=True))
        self.assertEqual(edges, {(0, 1), (1, 0), (2, 3), (3, 2)})

    def test_collation_keeps_molecule_and_label_alignment(self) -> None:
        batch = [
            configuration([1, 1], -20.0, "H2/rxn"),
            configuration([8, 1, 1], -60.0, "H2O/rxn"),
        ]
        collated = collate_configurations(batch, cutoff=2.0)
        self.assertEqual(collated["atomic_numbers"].shape, (5,))
        self.assertEqual(collated["positions"].shape, (5, 3))
        self.assertEqual(collated["forces"].shape, (5, 3))
        torch.testing.assert_close(
            collated["idx_m"], torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        )
        torch.testing.assert_close(
            collated["energy"], torch.tensor([-20.0, -60.0], dtype=torch.float64)
        )
        self.assertEqual(collated["reaction_ids"], ["H2/rxn", "H2O/rxn"])

    def test_relative_energy_ignores_one_constant_offset_per_reaction(self) -> None:
        targets = np.asarray([0.0, 1.0, 3.0, 5.0, 4.0])
        predictions = np.asarray([10.0, 11.0, 13.0, -2.0, -3.0])
        reaction_ids = ["a", "a", "a", "b", "b"]
        self.assertEqual(
            reaction_relative_rmse(predictions, targets, reaction_ids), 0.0
        )


if __name__ == "__main__":
    unittest.main()
