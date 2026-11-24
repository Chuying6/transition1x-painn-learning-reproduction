#!/usr/bin/env python3
"""Train SchNetPack's official PaiNN on mini-Transition1x.

This compact training loop is deliberately explicit for code review.  It uses
only the official training split to fit elemental energy baselines, derives
forces from the learned energy, and reports both absolute and reaction-relative
energy errors on the official validation and test formulas.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import schnetpack as spk
from schnetpack import properties

from inspect_t1x import ENERGY_DATASET, FORCES_DATASET


ELEMENTS = (1, 6, 7, 8)  # H, C, N, O
ELEMENT_SET = frozenset(ELEMENTS)


@dataclass
class Configuration:
    atomic_numbers: torch.Tensor
    positions: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor
    reaction_id: str


class InMemoryTransition1x(Dataset[Configuration]):
    """Load one mini split into memory; the full 6.6 GB file is not loaded."""

    def __init__(self, path: Path, split: str) -> None:
        self.configurations: list[Configuration] = []
        with h5py.File(path, "r") as handle:
            for formula in sorted(handle[split].keys()):
                formula_group = handle[split][formula]
                for reaction_name in sorted(formula_group.keys()):
                    reaction = formula_group[reaction_name]
                    atomic_numbers = torch.from_numpy(
                        reaction["atomic_numbers"][:].astype(np.int64)
                    )
                    positions = reaction["positions"][:].astype(np.float32)
                    energies = reaction[ENERGY_DATASET][:].astype(np.float64)
                    forces = reaction[FORCES_DATASET][:].astype(np.float32)
                    reaction_id = f"{formula}/{reaction_name}"
                    for frame in range(len(energies)):
                        self.configurations.append(
                            Configuration(
                                atomic_numbers=atomic_numbers.clone(),
                                positions=torch.from_numpy(positions[frame]),
                                energy=torch.tensor(energies[frame], dtype=torch.float64),
                                forces=torch.from_numpy(forces[frame]),
                                reaction_id=reaction_id,
                            )
                        )

    def __len__(self) -> int:
        return len(self.configurations)

    def __getitem__(self, index: int) -> Configuration:
        return self.configurations[index]


def composition(atomic_numbers: torch.Tensor) -> np.ndarray:
    values = atomic_numbers.numpy()
    unknown = sorted(set(int(value) for value in values) - ELEMENT_SET)
    if unknown:
        raise ValueError(f"unsupported atomic numbers in baseline fit: {unknown}")
    return np.asarray([(values == element).sum() for element in ELEMENTS], dtype=float)


def fit_element_baselines(dataset: InMemoryTransition1x) -> tuple[np.ndarray, float]:
    """Fit H/C/N/O energy offsets using training formulas only."""

    grouped_energies: dict[tuple[int, ...], list[float]] = {}
    for item in dataset.configurations:
        key = tuple(int(value) for value in composition(item.atomic_numbers))
        grouped_energies.setdefault(key, []).append(item.energy.item())

    design = np.asarray(list(grouped_energies.keys()), dtype=float)
    targets = np.asarray(
        [np.mean(grouped_energies[key]) for key in grouped_energies], dtype=float
    )
    baselines, _, rank, singular_values = np.linalg.lstsq(design, targets, rcond=None)
    if rank < len(ELEMENTS):
        raise ValueError(
            "training compositions cannot determine one baseline for every element"
        )
    condition_number = float(singular_values[0] / singular_values[-1])
    return baselines, condition_number


def full_neighbor_list(positions: list[torch.Tensor], cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build directed, non-self edges inside the cutoff for a molecular batch."""

    idx_i_parts: list[torch.Tensor] = []
    idx_j_parts: list[torch.Tensor] = []
    offset = 0
    for molecule_positions in positions:
        atom_count = molecule_positions.shape[0]
        distances = torch.cdist(molecule_positions, molecule_positions)
        mask = (distances < cutoff) & ~torch.eye(
            atom_count, dtype=torch.bool, device=distances.device
        )
        local_i, local_j = torch.where(mask)
        idx_i_parts.append(local_i + offset)
        idx_j_parts.append(local_j + offset)
        offset += atom_count
    return torch.cat(idx_i_parts), torch.cat(idx_j_parts)


def collate_configurations(batch: list[Configuration], cutoff: float) -> dict[str, object]:
    positions_list = [item.positions for item in batch]
    idx_i, idx_j = full_neighbor_list(positions_list, cutoff)
    return {
        "atomic_numbers": torch.cat([item.atomic_numbers for item in batch]),
        "positions": torch.cat(positions_list),
        "idx_i": idx_i,
        "idx_j": idx_j,
        "idx_m": torch.cat(
            [
                torch.full((len(item.atomic_numbers),), index, dtype=torch.long)
                for index, item in enumerate(batch)
            ]
        ),
        "energy": torch.stack([item.energy for item in batch]),
        "forces": torch.cat([item.forces for item in batch]),
        "reaction_ids": [item.reaction_id for item in batch],
        "batch_size": len(batch),
    }


class Transition1xPaiNN(nn.Module):
    def __init__(
        self,
        baselines: np.ndarray,
        cutoff: float,
        features: int,
        interactions: int,
    ) -> None:
        super().__init__()
        radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
        cutoff_function = spk.nn.CosineCutoff(cutoff)
        self.painn = spk.representation.PaiNN(
            n_atom_basis=features,
            n_interactions=interactions,
            radial_basis=radial_basis,
            cutoff_fn=cutoff_function,
        )
        self.energy_head = nn.Sequential(
            nn.Linear(features, features), nn.SiLU(), nn.Linear(features, 1)
        )

        lookup = torch.zeros(119, dtype=torch.float64)
        for atomic_number, baseline in zip(ELEMENTS, baselines):
            lookup[atomic_number] = float(baseline)
        self.register_buffer("element_baselines", lookup)

    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        atomic_numbers = batch["atomic_numbers"]
        positions = batch["positions"]
        idx_i = batch["idx_i"]
        idx_j = batch["idx_j"]
        idx_m = batch["idx_m"]
        batch_size = int(batch["batch_size"])

        inputs = {
            properties.Z: atomic_numbers,
            properties.R: positions,
            properties.Rij: positions[idx_j] - positions[idx_i],
            properties.idx_i: idx_i,
            properties.idx_j: idx_j,
            properties.idx_m: idx_m,
        }
        representation = self.painn(inputs)["scalar_representation"]
        atomic_residuals = self.energy_head(representation).squeeze(-1).to(torch.float64)
        atomic_energies = atomic_residuals + self.element_baselines[atomic_numbers]

        energies = torch.zeros(batch_size, dtype=torch.float64, device=positions.device)
        energies.index_add_(0, idx_m, atomic_energies)
        return energies


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def predict(
    model: nn.Module, batch: dict[str, object], create_graph: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    batch["positions"] = batch["positions"].detach().requires_grad_(True)
    energies = model(batch)
    forces = -torch.autograd.grad(
        energies.sum(), batch["positions"], create_graph=create_graph
    )[0]
    return energies, forces


def reaction_relative_rmse(
    predictions: np.ndarray, targets: np.ndarray, reaction_ids: list[str]
) -> float:
    squared_errors: list[np.ndarray] = []
    for reaction_id in sorted(set(reaction_ids)):
        indices = np.asarray([name == reaction_id for name in reaction_ids])
        predicted_relative = predictions[indices] - predictions[indices][0]
        target_relative = targets[indices] - targets[indices][0]
        squared_errors.append((predicted_relative - target_relative) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(squared_errors))))


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    predicted_energies: list[np.ndarray] = []
    target_energies: list[np.ndarray] = []
    predicted_forces: list[np.ndarray] = []
    target_forces: list[np.ndarray] = []
    reaction_ids: list[str] = []

    for host_batch in loader:
        batch = move_batch(host_batch, device)
        energies, forces = predict(model, batch, create_graph=False)
        predicted_energies.append(energies.detach().cpu().numpy())
        target_energies.append(batch["energy"].cpu().numpy())
        predicted_forces.append(forces.detach().cpu().numpy())
        target_forces.append(batch["forces"].cpu().numpy())
        reaction_ids.extend(batch["reaction_ids"])

    energy_prediction = np.concatenate(predicted_energies)
    energy_target = np.concatenate(target_energies)
    force_prediction = np.concatenate(predicted_forces)
    force_target = np.concatenate(target_forces)
    return {
        "energy_rmse": float(np.sqrt(np.mean((energy_prediction - energy_target) ** 2))),
        "relative_energy_rmse": reaction_relative_rmse(
            energy_prediction, energy_target, reaction_ids
        ),
        "force_rmse": float(np.sqrt(np.mean((force_prediction - force_target) ** 2))),
    }


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument(
        "--eval-hdf5",
        type=Path,
        default=None,
        help="Optional file supplying fixed val/test splits for controlled comparisons",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--interactions", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mini_painn.pt"))
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional path for saving epoch history and final test metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"batch size must be positive, got {args.batch_size}")
    if args.cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {args.cutoff}")
    if args.features < 1 or args.interactions < 1:
        raise ValueError("features and interactions must both be positive")
    if args.learning_rate <= 0.0:
        raise ValueError(f"learning rate must be positive, got {args.learning_rate}")
    if args.seed < 0:
        raise ValueError(f"seed must be non-negative, got {args.seed}")
    if not args.hdf5.is_file():
        raise FileNotFoundError(f"training HDF5 does not exist: {args.hdf5}")
    if args.eval_hdf5 is not None and not args.eval_hdf5.is_file():
        raise FileNotFoundError(f"evaluation HDF5 does not exist: {args.eval_hdf5}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    eval_hdf5 = args.eval_hdf5 or args.hdf5
    train_data = InMemoryTransition1x(args.hdf5, "train")
    val_data = InMemoryTransition1x(eval_hdf5, "val")
    test_data = InMemoryTransition1x(eval_hdf5, "test")
    baselines, condition_number = fit_element_baselines(train_data)

    collate = lambda batch: collate_configurations(batch, args.cutoff)  # noqa: E731
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    model = Transition1xPaiNN(
        baselines, args.cutoff, args.features, args.interactions
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val = math.inf
    best_epoch = 0
    history: list[dict[str, float | int]] = []

    print(f"device: {device}", flush=True)
    print(
        f"train/val/test: {len(train_data)}/{len(val_data)}/{len(test_data)}",
        flush=True,
    )
    print(f"baseline condition number: {condition_number:.2f}", flush=True)
    print(
        "element baselines (eV): "
        + ", ".join(
            f"Z={element}: {value:.6f}"
            for element, value in zip(ELEMENTS, baselines)
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        energy_squared_error = 0.0
        force_squared_error = 0.0
        energy_count = 0
        force_count = 0

        for host_batch in train_loader:
            batch = move_batch(host_batch, device)
            optimizer.zero_grad()
            energies, forces = predict(model, batch, create_graph=True)
            energy_loss = torch.mean((energies - batch["energy"]) ** 2)
            force_loss = torch.mean((forces - batch["forces"]) ** 2)
            loss = 0.01 * energy_loss + 0.99 * force_loss
            loss.backward()
            optimizer.step()

            energy_squared_error += energy_loss.item() * int(batch["batch_size"])
            energy_count += int(batch["batch_size"])
            force_squared_error += force_loss.item() * batch["forces"].numel()
            force_count += batch["forces"].numel()

        train_energy_rmse = math.sqrt(energy_squared_error / energy_count)
        train_force_rmse = math.sqrt(force_squared_error / force_count)
        val_metrics = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_energy_rmse_eV": train_energy_rmse,
                "train_force_rmse_eV_per_A": train_force_rmse,
                "val_energy_rmse_eV": val_metrics["energy_rmse"],
                "val_relative_energy_rmse_eV": val_metrics[
                    "relative_energy_rmse"
                ],
                "val_force_rmse_eV_per_A": val_metrics["force_rmse"],
            }
        )
        print(
            f"epoch {epoch:3d} | train E {train_energy_rmse:.4f} | "
            f"train F {train_force_rmse:.4f} | val E {val_metrics['energy_rmse']:.4f} | "
            f"val relative E {val_metrics['relative_energy_rmse']:.4f} | "
            f"val F {val_metrics['force_rmse']:.4f}",
            flush=True,
        )

        if val_metrics["force_rmse"] < best_val:
            best_val = val_metrics["force_rmse"]
            best_epoch = epoch
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint)

    if best_epoch == 0:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, device)
    print("\nBest-checkpoint test metrics")
    print(f"absolute energy RMSE : {test_metrics['energy_rmse']:.6f} eV")
    print(f"relative energy RMSE : {test_metrics['relative_energy_rmse']:.6f} eV")
    print(f"force RMSE           : {test_metrics['force_rmse']:.6f} eV/A")

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "training_hdf5": str(args.hdf5),
            "evaluation_hdf5": str(eval_hdf5),
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "seed": args.seed,
            "dataset_sizes": {
                "train": len(train_data),
                "validation": len(val_data),
                "test": len(test_data),
            },
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "cutoff_A": args.cutoff,
                "features": args.features,
                "interactions": args.interactions,
                "learning_rate": args.learning_rate,
                "energy_loss_weight": 0.01,
                "force_loss_weight": 0.99,
            },
            "element_baselines_eV": {
                str(element): float(value)
                for element, value in zip(ELEMENTS, baselines)
            },
            "baseline_condition_number": condition_number,
            "versions": {
                "torch": torch.__version__,
                "schnetpack": importlib.metadata.version("schnetpack"),
                "numpy": np.__version__,
                "h5py": h5py.__version__,
            },
            "best_epoch_by_val_force": best_epoch,
            "history": history,
            "test": {
                "energy_rmse_eV": test_metrics["energy_rmse"],
                "relative_energy_rmse_eV": test_metrics[
                    "relative_energy_rmse"
                ],
                "force_rmse_eV_per_A": test_metrics["force_rmse"],
            },
        }
        args.metrics_json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"metrics JSON          : {args.metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
