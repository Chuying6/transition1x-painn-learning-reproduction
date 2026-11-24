#!/usr/bin/env python3
"""Build a chemistry-diverse mini-Transition1x from the official splits.

Unlike the narrow first experiment, this subset includes every chemical
formula in each official split.  It selects only a small number of reactions
per formula and a small number of evenly spaced frames per reaction.  Formula
leakage remains impossible because formulas never move between official splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from build_mini_t1x import copy_reaction, evenly_spaced_indices
from inspect_t1x import ENERGY_DATASET


SPLITS = ("train", "val", "test")


def choose_reactions(
    reaction_names: list[str], count: int, rng: np.random.Generator
) -> list[str]:
    """Choose reaction names reproducibly without depending on HDF5 order."""

    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    names = sorted(reaction_names)
    if not names:
        raise ValueError("cannot sample from an empty reaction list")
    if len(names) <= count:
        return names
    chosen_indices = np.sort(rng.choice(len(names), size=count, replace=False))
    return [names[int(index)] for index in chosen_indices]


def build_diverse_subset(
    source_path: Path,
    output_path: Path,
    reactions_per_formula: int,
    frames_per_reaction: int,
    seed: int,
) -> dict[str, object]:
    if reactions_per_formula < 1 or frames_per_reaction < 1:
        raise ValueError("sampling counts must both be positive")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if output_path.exists():
        raise FileExistsError(
            f"output already exists: {output_path}; remove it explicitly to rebuild"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    summary: dict[str, object] = {
        "source": str(source_path),
        "seed": seed,
        "reactions_per_formula": reactions_per_formula,
        "frames_per_reaction": frames_per_reaction,
        "splits": {},
    }

    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as output:
        data_group = output.create_group("data")
        output.attrs["source_file"] = str(source_path)
        output.attrs["sampling_strategy"] = (
            "all official formulas; reactions sampled without replacement from one "
            "global PCG64 stream; evenly spaced frames per reaction"
        )
        output.attrs["reaction_sampling_version"] = "global_choice_v1"
        output.attrs["seed"] = seed
        output.attrs["reactions_per_formula"] = reactions_per_formula
        output.attrs["frames_per_reaction"] = frames_per_reaction

        for split in SPLITS:
            split_group = output.create_group(split)
            formula_count = 0
            reaction_count = 0
            frame_count = 0

            for formula in sorted(source[split].keys()):
                source_formula = source[split][formula]
                destination_formula = data_group.create_group(formula)
                chosen_reactions = choose_reactions(
                    list(source_formula.keys()), reactions_per_formula, rng
                )

                for reaction_name in chosen_reactions:
                    source_reaction = source_formula[reaction_name]
                    total_frames = source_reaction[ENERGY_DATASET].shape[0]
                    frame_indices = evenly_spaced_indices(
                        total_frames, frames_per_reaction
                    )
                    destination_reaction = destination_formula.create_group(
                        reaction_name
                    )
                    copy_reaction(
                        source_reaction, destination_reaction, frame_indices
                    )
                    reaction_count += 1
                    frame_count += int(frame_indices.size)

                split_group[formula] = h5py.SoftLink(f"/data/{formula}")
                formula_count += 1

            summary["splits"][split] = {
                "formulas": formula_count,
                "reactions": reaction_count,
                "frames": frame_count,
            }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reactions-per-formula", type=int, default=2)
    parser.add_argument("--frames-per-reaction", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"source file does not exist: {args.source}")
    summary = build_diverse_subset(
        args.source,
        args.output,
        args.reactions_per_formula,
        args.frames_per_reaction,
        args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
