#!/usr/bin/env python3
"""Export component-wise test-force predictions from a trained PaiNN checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_mini_t1x_painn import (  # noqa: E402
    InMemoryTransition1x,
    Transition1xPaiNN,
    choose_device,
    collate_configurations,
    fit_element_baselines,
    move_batch,
    predict,
)


COMPONENTS = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-hdf5",
        type=Path,
        default=PROJECT_ROOT / "data" / "mini_expanded_transition1x.h5",
        help="training split used to reconstruct elemental energy baselines",
    )
    parser.add_argument(
        "--evaluation-hdf5",
        type=Path,
        default=PROJECT_ROOT / "data" / "mini_diverse_equal_transition1x.h5",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "expanded_mini_painn_complete.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "expanded_test_force_predictions.csv",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--interactions", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.training_hdf5, args.evaluation_hdf5, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    device = choose_device(args.device)
    train_data = InMemoryTransition1x(args.training_hdf5, "train")
    test_data = InMemoryTransition1x(args.evaluation_hdf5, "test")
    baselines, _ = fit_element_baselines(train_data)
    collate = lambda batch: collate_configurations(batch, args.cutoff)  # noqa: E731
    loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = Transition1xPaiNN(
        baselines, args.cutoff, args.features, args.interactions
    ).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "configuration_index",
        "reaction_id",
        "atomic_number",
        "atom_index",
        "component",
        "reference_force_eV_per_A",
        "predicted_force_eV_per_A",
        "residual_eV_per_A",
    )
    configuration_index = 0
    row_count = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for host_batch in loader:
            batch = move_batch(host_batch, device)
            _, predicted_forces = predict(model, batch, create_graph=False)
            reference_forces = batch["forces"]
            atomic_numbers = batch["atomic_numbers"]
            molecule_indices = batch["idx_m"]

            for local_molecule, reaction_id in enumerate(batch["reaction_ids"]):
                atom_mask = molecule_indices == local_molecule
                molecule_prediction = predicted_forces[atom_mask].detach().cpu()
                molecule_reference = reference_forces[atom_mask].detach().cpu()
                molecule_numbers = atomic_numbers[atom_mask].detach().cpu()
                for atom_index, atomic_number in enumerate(molecule_numbers.tolist()):
                    for component_index, component in enumerate(COMPONENTS):
                        reference = float(
                            molecule_reference[atom_index, component_index]
                        )
                        prediction = float(
                            molecule_prediction[atom_index, component_index]
                        )
                        writer.writerow(
                            {
                                "configuration_index": configuration_index,
                                "reaction_id": reaction_id,
                                "atomic_number": atomic_number,
                                "atom_index": atom_index,
                                "component": component,
                                "reference_force_eV_per_A": f"{reference:.9g}",
                                "predicted_force_eV_per_A": f"{prediction:.9g}",
                                "residual_eV_per_A": f"{prediction - reference:.9g}",
                            }
                        )
                        row_count += 1
                configuration_index += 1

    print(f"device: {device}")
    print(f"test configurations: {configuration_index}")
    print(f"force components: {row_count}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
