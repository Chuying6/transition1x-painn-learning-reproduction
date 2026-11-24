#!/usr/bin/env python3
"""Build a deterministic, leakage-free mini subset of Transition1x.

The official train/validation/test split is defined by chemical formula.  This
script selects different formulas from each official split, concatenates the
ordered frames of all reactions within each formula, and samples evenly from
that formula-level frame pool.  Longer reactions therefore contribute more
frames.  The reaction-group hierarchy and method-qualified energy/force dataset
names are preserved.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from inspect_t1x import ENERGY_DATASET, FORCES_DATASET


DEFAULT_SELECTION: dict[str, dict[str, int]] = {
    "train": {
        "CH4N2O": 750,
        "C3HN3O": 750,
        "C3H2N2O": 750,
        "C2H3NO": 750,
    },
    "val": {"C2H6N2O": 500},
    "test": {"C2H3N3O2": 500},
}

ENDPOINT_GROUPS = ("reactant", "transition_state", "product")


def evenly_spaced_indices(total: int, limit: int) -> np.ndarray:
    """Return at most ``limit`` deterministic indices spanning ``total`` items."""

    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    if total <= limit:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=limit, dtype=np.int64)


def selected_frames(
    formula_group: h5py.Group, limit: int
) -> dict[str, np.ndarray]:
    """Sample one formula-level frame pool and regroup by reaction name."""

    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")

    locations: list[tuple[str, int]] = []
    for reaction_name in sorted(formula_group.keys()):
        reaction = formula_group[reaction_name]
        frame_count = reaction[ENERGY_DATASET].shape[0]
        locations.extend((reaction_name, frame) for frame in range(frame_count))

    if not locations:
        raise ValueError(f"formula group {formula_group.name!r} contains no frames")

    chosen: defaultdict[str, list[int]] = defaultdict(list)
    for index in evenly_spaced_indices(len(locations), limit):
        reaction_name, frame = locations[int(index)]
        chosen[reaction_name].append(frame)

    return {
        reaction_name: np.asarray(frames, dtype=np.int64)
        for reaction_name, frames in chosen.items()
    }


def copy_endpoint_groups(source: h5py.Group, destination: h5py.Group) -> None:
    for name in ENDPOINT_GROUPS:
        if name in source:
            source.copy(name, destination, name=name)


def copy_reaction(
    source: h5py.Group, destination: h5py.Group, frame_indices: np.ndarray
) -> None:
    destination.attrs["source_frame_indices"] = frame_indices
    destination.create_dataset("atomic_numbers", data=source["atomic_numbers"][:])
    destination.create_dataset("positions", data=source["positions"][frame_indices])
    destination.create_dataset(ENERGY_DATASET, data=source[ENERGY_DATASET][frame_indices])
    destination.create_dataset(FORCES_DATASET, data=source[FORCES_DATASET][frame_indices])
    copy_endpoint_groups(source, destination)


def build_subset(source_path: Path, output_path: Path) -> dict[str, object]:
    """Create the mini HDF5 file and return a JSON-serializable summary."""

    if output_path.exists():
        raise FileExistsError(
            f"output already exists: {output_path}; remove it explicitly to rebuild"
        )

    summary: dict[str, object] = {"source": str(source_path), "splits": {}}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as output:
        data_group = output.create_group("data")
        output.attrs["source_file"] = str(source_path)
        output.attrs["selection"] = json.dumps(DEFAULT_SELECTION, sort_keys=True)
        output.attrs["sampling_strategy"] = (
            "formula-level ordered frame pool; deterministic evenly spaced indices"
        )

        for split, formulas in DEFAULT_SELECTION.items():
            split_group = output.create_group(split)
            split_summary: dict[str, object] = {}

            for formula, limit in formulas.items():
                if formula not in source[split]:
                    raise KeyError(f"{formula!r} is not present in official split {split!r}")
                if formula in data_group:
                    raise ValueError(f"formula leakage in requested subset: {formula}")

                source_formula = source[split][formula]
                destination_formula = data_group.create_group(formula)
                chosen = selected_frames(source_formula, limit)
                reaction_counts: dict[str, int] = {}

                for reaction_name, frame_indices in chosen.items():
                    source_reaction = source_formula[reaction_name]
                    destination_reaction = destination_formula.create_group(reaction_name)
                    copy_reaction(source_reaction, destination_reaction, frame_indices)
                    reaction_counts[reaction_name] = int(frame_indices.size)

                split_group[formula] = h5py.SoftLink(f"/data/{formula}")
                split_summary[formula] = {
                    "frames": int(sum(reaction_counts.values())),
                    "reactions": len(reaction_counts),
                    "reaction_frames": reaction_counts,
                }

            summary["splits"][split] = split_summary

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"source file does not exist: {args.source}")
    summary = build_subset(args.source, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
