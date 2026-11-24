from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_diverse_mini_t1x import (  # noqa: E402
    build_diverse_subset,
    choose_reactions,
)
from inspect_t1x import ENERGY_DATASET, FORCES_DATASET, inspect_file  # noqa: E402


def add_reaction(group: h5py.Group, name: str, frames: int = 7) -> None:
    reaction = group.create_group(name)
    reaction.create_dataset("atomic_numbers", data=np.asarray([1, 8]))
    positions = np.arange(frames * 2 * 3, dtype=np.float64).reshape(frames, 2, 3)
    reaction.create_dataset("positions", data=positions)
    reaction.create_dataset(ENERGY_DATASET, data=np.arange(frames, dtype=np.float64))
    reaction.create_dataset(FORCES_DATASET, data=-positions)


class DiverseSubsetBuilderTests(unittest.TestCase):
    def test_reaction_choice_is_reproducible_and_order_independent(self) -> None:
        names = ["rxn_3", "rxn_1", "rxn_4", "rxn_2"]
        first = choose_reactions(names, 2, np.random.default_rng(7))
        second = choose_reactions(list(reversed(names)), 2, np.random.default_rng(7))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(set(first)), 2)

    def test_invalid_sampling_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            choose_reactions(["rxn"], 0, np.random.default_rng(7))
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "positive"):
                build_diverse_subset(
                    Path(temp_dir) / "source.h5",
                    Path(temp_dir) / "output.h5",
                    reactions_per_formula=0,
                    frames_per_reaction=10,
                    seed=7,
                )

    def test_builder_preserves_splits_and_reaction_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.h5"
            output_path = Path(temp_dir) / "output.h5"
            with h5py.File(source_path, "w") as source:
                data = source.create_group("data")
                for split, formula in (
                    ("train", "H2O"),
                    ("val", "H3N"),
                    ("test", "CH4"),
                ):
                    formula_group = data.create_group(formula)
                    for index in range(3):
                        add_reaction(formula_group, f"rxn_{index}", frames=7)
                    split_group = source.create_group(split)
                    split_group[formula] = h5py.SoftLink(f"/data/{formula}")

            summary = build_diverse_subset(
                source_path,
                output_path,
                reactions_per_formula=2,
                frames_per_reaction=4,
                seed=7,
            )
            self.assertEqual(inspect_file(output_path)["errors"], [])
            for split in ("train", "val", "test"):
                self.assertEqual(summary["splits"][split]["formulas"], 1)
                self.assertEqual(summary["splits"][split]["reactions"], 2)
                self.assertEqual(summary["splits"][split]["frames"], 8)

            with h5py.File(output_path, "r") as output:
                self.assertEqual(
                    output.attrs["reaction_sampling_version"], "global_choice_v1"
                )
                for split in ("train", "val", "test"):
                    formula = next(iter(output[split].keys()))
                    for reaction in output[split][formula].values():
                        np.testing.assert_array_equal(
                            reaction.attrs["source_frame_indices"],
                            np.asarray([0, 2, 4, 6]),
                        )


if __name__ == "__main__":
    unittest.main()
