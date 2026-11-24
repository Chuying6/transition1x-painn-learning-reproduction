from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_mini_t1x import (  # noqa: E402
    DEFAULT_SELECTION,
    build_subset,
    copy_reaction,
    evenly_spaced_indices,
    selected_frames,
)
from inspect_t1x import ENERGY_DATASET, FORCES_DATASET, inspect_file  # noqa: E402


def add_reaction(group: h5py.Group, name: str, frames: int, atoms: int = 2) -> None:
    reaction = group.create_group(name)
    reaction.create_dataset("atomic_numbers", data=np.arange(1, atoms + 1))
    positions = np.arange(frames * atoms * 3, dtype=np.float64).reshape(frames, atoms, 3)
    reaction.create_dataset("positions", data=positions)
    reaction.create_dataset(ENERGY_DATASET, data=np.arange(frames, dtype=np.float64))
    reaction.create_dataset(FORCES_DATASET, data=-positions)
    for endpoint in ("reactant", "transition_state", "product"):
        endpoint_group = reaction.create_group(endpoint)
        endpoint_group.create_dataset("marker", data=np.asarray([frames]))


class MiniSubsetBuilderTests(unittest.TestCase):
    def test_evenly_spaced_indices_are_deterministic_and_unique(self) -> None:
        np.testing.assert_array_equal(
            evenly_spaced_indices(total=10, limit=4), np.asarray([0, 3, 6, 9])
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            evenly_spaced_indices(total=10, limit=0)

    def test_formula_pool_sampling_is_regrouped_by_reaction(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".h5") as temp:
            with h5py.File(temp.name, "w") as handle:
                formula = handle.create_group("formula")
                add_reaction(formula, "rxn_a", frames=3)
                add_reaction(formula, "rxn_b", frames=7)
                chosen = selected_frames(formula, limit=4)
                np.testing.assert_array_equal(chosen["rxn_a"], np.asarray([0]))
                np.testing.assert_array_equal(chosen["rxn_b"], np.asarray([0, 3, 6]))

    def test_copy_reaction_keeps_arrays_aligned_and_records_provenance(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".h5") as temp:
            with h5py.File(temp.name, "w") as handle:
                source = handle.create_group("source")
                destination = handle.create_group("destination")
                add_reaction(source, "rxn", frames=5)
                indices = np.asarray([0, 2, 4], dtype=np.int64)
                copy_reaction(source["rxn"], destination, indices)

                np.testing.assert_array_equal(
                    destination.attrs["source_frame_indices"], indices
                )
                np.testing.assert_array_equal(
                    destination[ENERGY_DATASET][:], np.asarray([0.0, 2.0, 4.0])
                )
                np.testing.assert_array_equal(
                    destination["positions"][:], source["rxn/positions"][indices]
                )
                np.testing.assert_array_equal(
                    destination[FORCES_DATASET][:], source["rxn"][FORCES_DATASET][indices]
                )
                self.assertIn("transition_state", destination)

    def test_complete_subset_preserves_official_formula_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.h5"
            output_path = Path(temp_dir) / "mini.h5"
            with h5py.File(source_path, "w") as source:
                data = source.create_group("data")
                for split, formulas in DEFAULT_SELECTION.items():
                    split_group = source.create_group(split)
                    for formula in formulas:
                        formula_group = data.create_group(formula)
                        add_reaction(formula_group, "rxn_000", frames=3)
                        split_group[formula] = h5py.SoftLink(f"/data/{formula}")

            summary = build_subset(source_path, output_path)
            self.assertEqual(inspect_file(output_path)["errors"], [])
            with h5py.File(output_path, "r") as output:
                selection = json.loads(output.attrs["selection"])
                self.assertEqual(selection, DEFAULT_SELECTION)
                self.assertIn("formula-level", output.attrs["sampling_strategy"])
                for split, formulas in DEFAULT_SELECTION.items():
                    self.assertEqual(set(output[split].keys()), set(formulas))
                    for formula in formulas:
                        self.assertIsInstance(
                            output[split].get(formula, getlink=True), h5py.SoftLink
                        )
            self.assertEqual(set(summary["splits"]), set(DEFAULT_SELECTION))


if __name__ == "__main__":
    unittest.main()
