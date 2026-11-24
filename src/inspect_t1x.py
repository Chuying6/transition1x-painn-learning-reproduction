#!/usr/bin/env python3
"""Inspect the Transition1x HDF5 schema and split integrity.

This is the first, model-free stage of the Transition1x reproduction.  It checks
the structural claims made in the paper without loading full coordinate arrays
into memory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py


ENERGY_DATASET = "wB97x_6-31G(d).energy"
FORCES_DATASET = "wB97x_6-31G(d).forces"
CORE_DATASETS = ("atomic_numbers", ENERGY_DATASET, FORCES_DATASET, "positions")
SPLITS = ("train", "val", "test")


@dataclass
class SplitSummary:
    formulas: int = 0
    reactions: int = 0
    configurations: int = 0
    soft_links: int = 0
    external_links: int = 0
    hard_links: int = 0


def _link_kind(group: h5py.Group, name: str) -> str:
    link = group.get(name, getlink=True)
    if isinstance(link, h5py.SoftLink):
        return "soft"
    if isinstance(link, h5py.ExternalLink):
        return "external"
    return "hard"


def _validate_reaction(path: str, group: h5py.Group) -> tuple[int, list[str]]:
    errors: list[str] = []
    missing = [name for name in CORE_DATASETS if name not in group]
    if missing:
        return 0, [f"{path}: missing datasets {missing}"]

    z_shape = group["atomic_numbers"].shape
    e_shape = group[ENERGY_DATASET].shape
    f_shape = group[FORCES_DATASET].shape
    r_shape = group["positions"].shape

    if len(z_shape) != 1:
        errors.append(f"{path}/atomic_numbers: expected (m,), got {z_shape}")
        return 0, errors
    m = z_shape[0]

    if len(e_shape) != 1:
        errors.append(f"{path}/{ENERGY_DATASET}: expected (n,), got {e_shape}")
        return 0, errors
    n = e_shape[0]

    expected_vector_shape = (n, m, 3)
    if f_shape != expected_vector_shape:
        errors.append(
            f"{path}/{FORCES_DATASET}: expected {expected_vector_shape}, got {f_shape}"
        )
    if r_shape != expected_vector_shape:
        errors.append(
            f"{path}/positions: expected {expected_vector_shape}, got {r_shape}"
        )
    return n, errors


def inspect_file(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "top_level_groups": [],
        "splits": {},
        "formula_overlaps": {},
        "errors": [],
    }

    with h5py.File(path, "r") as handle:
        report["top_level_groups"] = sorted(handle.keys())
        required = {"data", *SPLITS}
        missing_top = sorted(required - set(handle.keys()))
        if missing_top:
            report["errors"].append(f"missing top-level groups: {missing_top}")

        formula_sets: dict[str, set[str]] = {}
        for split in SPLITS:
            if split not in handle:
                formula_sets[split] = set()
                continue

            split_group = handle[split]
            formulas = set(split_group.keys())
            formula_sets[split] = formulas
            summary = SplitSummary(formulas=len(formulas))

            for formula in sorted(formulas):
                link_kind = _link_kind(split_group, formula)
                setattr(summary, f"{link_kind}_links", getattr(summary, f"{link_kind}_links") + 1)

                try:
                    formula_group = split_group[formula]
                except (KeyError, OSError) as exc:
                    report["errors"].append(
                        f"/{split}/{formula}: cannot resolve {link_kind} link: {exc}"
                    )
                    continue
                if not isinstance(formula_group, h5py.Group):
                    report["errors"].append(
                        f"/{split}/{formula}: expected a formula group, "
                        f"got {type(formula_group).__name__}"
                    )
                    continue
                for reaction_name, item in formula_group.items():
                    if not isinstance(item, h5py.Group):
                        continue
                    if not any(name in item for name in CORE_DATASETS):
                        continue
                    reaction_path = f"/{split}/{formula}/{reaction_name}"
                    n_configs, errors = _validate_reaction(reaction_path, item)
                    summary.reactions += 1
                    summary.configurations += n_configs
                    report["errors"].extend(errors)

            report["splits"][split] = asdict(summary)

        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sorted(formula_sets[left] & formula_sets[right])
            key = f"{left}_vs_{right}"
            report["formula_overlaps"][key] = overlap
            if overlap:
                report["errors"].append(
                    f"formula leakage between {left} and {right}: {overlap[:10]}"
                )

    return report


def _print_human(report: dict[str, Any]) -> None:
    print(f"File: {report['path']}")
    print("Top-level groups:", ", ".join(report["top_level_groups"]))
    print()
    print(
        f"{'split':<8} {'formula':>8} {'reaction':>10} {'config':>10} "
        f"{'soft':>6} {'external':>9} {'hard':>6}"
    )
    for split in SPLITS:
        values = report["splits"].get(split, {})
        print(
            f"{split:<8} "
            f"{values.get('formulas', 0):>8} "
            f"{values.get('reactions', 0):>10} "
            f"{values.get('configurations', 0):>10} "
            f"{values.get('soft_links', 0):>6} "
            f"{values.get('external_links', 0):>9} "
            f"{values.get('hard_links', 0):>6}"
        )
    print()
    if report["errors"]:
        print("FAILED")
        for error in report["errors"]:
            print(f"- {error}")
    else:
        print("PASSED: schema shapes are consistent and formula splits do not overlap.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path, help="Path to transition1x.h5")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    args = parser.parse_args()

    if not args.hdf5.is_file():
        parser.error(f"file does not exist: {args.hdf5}")

    report = inspect_file(args.hdf5)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

