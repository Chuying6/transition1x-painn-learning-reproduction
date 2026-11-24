from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inspect_t1x import ENERGY_DATASET, FORCES_DATASET, inspect_file  # noqa: E402


def add_reaction(formula_group: h5py.Group, name: str, n: int, m: int) -> None:
    reaction = formula_group.create_group(name)
    reaction.create_dataset("atomic_numbers", data=np.ones((m,), dtype=np.int64))
    reaction.create_dataset(ENERGY_DATASET, data=np.zeros((n,), dtype=np.float64))
    reaction.create_dataset(FORCES_DATASET, data=np.zeros((n, m, 3), dtype=np.float64))
    reaction.create_dataset("positions", data=np.zeros((n, m, 3), dtype=np.float64))


class InspectTransition1xTests(unittest.TestCase):
    def make_file(self, duplicate_formula: bool = False) -> Path:
        temp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        temp.close()
        path = Path(temp.name)
        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            for formula, split, n, m in (
                ("H2", "train", 5, 2),
                ("H2O", "val", 7, 3),
                ("CH4", "test", 9, 5),
            ):
                formula_group = data.create_group(formula)
                add_reaction(formula_group, "rxn_000", n=n, m=m)
                handle.require_group(split)[formula] = h5py.SoftLink(f"/data/{formula}")
            for split in ("train", "val", "test"):
                handle.require_group(split)
            if duplicate_formula:
                handle["test"]["H2"] = h5py.SoftLink("/data/H2")
        return path

    def test_valid_schema_and_disjoint_splits(self) -> None:
        path = self.make_file()
        self.addCleanup(path.unlink)
        report = inspect_file(path)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["splits"]["train"]["configurations"], 5)
        self.assertEqual(report["splits"]["val"]["configurations"], 7)
        self.assertEqual(report["splits"]["test"]["configurations"], 9)

    def test_formula_leakage_is_reported(self) -> None:
        path = self.make_file(duplicate_formula=True)
        self.addCleanup(path.unlink)
        report = inspect_file(path)
        self.assertIn("H2", report["formula_overlaps"]["train_vs_test"])
        self.assertTrue(any("formula leakage" in error for error in report["errors"]))

    def test_invalid_force_shape_is_reported(self) -> None:
        path = self.make_file()
        self.addCleanup(path.unlink)
        with h5py.File(path, "r+") as handle:
            reaction = handle["data/H2/rxn_000"]
            del reaction[FORCES_DATASET]
            reaction.create_dataset(
                FORCES_DATASET, data=np.zeros((5, 2, 2), dtype=np.float64)
            )

        report = inspect_file(path)
        self.assertTrue(
            any(
                FORCES_DATASET in error and "expected (5, 2, 3)" in error
                for error in report["errors"]
            )
        )

    def test_broken_formula_link_is_reported(self) -> None:
        path = self.make_file()
        self.addCleanup(path.unlink)
        with h5py.File(path, "r+") as handle:
            handle["train"]["broken"] = h5py.SoftLink("/data/not_present")

        report = inspect_file(path)
        self.assertTrue(
            any("cannot resolve soft link" in error for error in report["errors"])
        )


if __name__ == "__main__":
    unittest.main()
