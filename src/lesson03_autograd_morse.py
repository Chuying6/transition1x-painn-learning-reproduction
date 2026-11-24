#!/usr/bin/env python3
"""Minimal energy-force training experiment with PyTorch autograd.

The network predicts a scalar energy E(q).  Force is not a second independent
output: it is constructed as F(q) = -dE/dq.  This is the same differentiation
pattern used by energy-conserving machine-learning interatomic potentials.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MorseParameters:
    dissociation_energy: float = 4.5  # eV
    inverse_length: float = 1.8  # 1/angstrom


def morse_energy(q: torch.Tensor, params: MorseParameters) -> torch.Tensor:
    """Reference Morse energy in eV."""
    return params.dissociation_energy * (
        1.0 - torch.exp(-params.inverse_length * q)
    ).square()


def morse_force(q: torch.Tensor, params: MorseParameters) -> torch.Tensor:
    """Analytical reference force, -dE/dq, in eV/angstrom."""
    exponential = torch.exp(-params.inverse_length * q)
    return (
        -2.0
        * params.dissociation_energy
        * params.inverse_length
        * exponential
        * (1.0 - exponential)
    )


class EnergyNetwork(nn.Module):
    """Small scalar neural network E_theta(q)."""

    def __init__(self, hidden_features: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.layers(q)


def energy_and_force(
    model: nn.Module,
    q: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate energy and construct the conservative force with autograd."""
    if not q.requires_grad:
        q = q.detach().clone().requires_grad_(True)
    energy = model(q)
    gradient = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=q,
        create_graph=create_graph,
    )[0]
    force = -gradient
    return energy, force


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    params = MorseParameters()

    # q is the displacement from equilibrium in angstrom. Training covers the
    # minimum and moderate stretching, while test data probe extrapolation.
    q_train = torch.linspace(-0.25, 0.60, 96, device=device, dtype=dtype).reshape(-1, 1)
    q_test = torch.linspace(0.61, 1.60, 96, device=device, dtype=dtype).reshape(-1, 1)

    e_train = morse_energy(q_train, params)
    f_train = morse_force(q_train, params)
    e_test = morse_energy(q_test, params)
    f_test = morse_force(q_test, params)
    return q_train, e_train, f_train, q_test, e_test, f_test


def train(
    *,
    mode: str,
    epochs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    if mode not in {"energy", "energy-force"}:
        raise ValueError(f"unsupported training mode: {mode}")
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")

    torch.manual_seed(7)
    model = EnergyNetwork().to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)
    q_train, e_train, f_train, _, _, _ = make_dataset(device, dtype)

    for epoch in range(1, epochs + 1):
        q_batch = q_train.detach().clone().requires_grad_(True)
        pred_e, pred_f = energy_and_force(model, q_batch, create_graph=True)

        energy_loss = torch.mean((pred_e - e_train).square())
        force_loss = torch.mean((pred_f - f_train).square())
        if mode == "energy":
            loss = energy_loss
        else:
            # The factor only balances numerical scales in this teaching example.
            loss = energy_loss + 0.05 * force_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 500 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:5d} total={loss.item():.6f} "
                f"energy={energy_loss.item():.6f} force={force_loss.item():.6f}"
            )
    return model


def rmse(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((predicted - reference).square())).item()


def evaluate(model: nn.Module, device: torch.device, dtype: torch.dtype) -> None:
    q_train, e_train, f_train, q_test, e_test, f_test = make_dataset(device, dtype)

    train_e, train_f = energy_and_force(model, q_train, create_graph=False)
    test_e, test_f = energy_and_force(model, q_test, create_graph=False)

    print("\nRMSE")
    print(f"train energy: {rmse(train_e, e_train):.6f} eV")
    print(f"train force : {rmse(train_f, f_train):.6f} eV/angstrom")
    print(f"test energy : {rmse(test_e, e_test):.6f} eV")
    print(f"test force  : {rmse(test_f, f_test):.6f} eV/angstrom")


def finite_difference_check(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    q_value: float = 0.40,
) -> None:
    """Check -dE/dq independently using a central finite difference."""
    step = 1.0e-3 if dtype == torch.float32 else 1.0e-5
    q = torch.tensor([[q_value]], device=device, dtype=dtype, requires_grad=True)
    _, force_autograd = energy_and_force(model, q, create_graph=False)

    with torch.no_grad():
        q_plus = torch.tensor([[q_value + step]], device=device, dtype=dtype)
        q_minus = torch.tensor([[q_value - step]], device=device, dtype=dtype)
        energy_plus = model(q_plus)
        energy_minus = model(q_minus)
        force_finite_difference = -(energy_plus - energy_minus) / (2.0 * step)

    difference = torch.abs(force_autograd - force_finite_difference).item()
    print("\nFinite-difference check")
    print(f"autograd force        : {force_autograd.item(): .8f} eV/angstrom")
    print(f"finite-difference force: {force_finite_difference.item(): .8f} eV/angstrom")
    print(f"absolute difference   : {difference:.3e} eV/angstrom")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("energy", "energy-force"),
        default="energy-force",
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    args = parser.parse_args()

    device = select_device(args.device)
    dtype = torch.float32 if device.type == "mps" else torch.float64
    print(f"device={device} dtype={dtype} mode={args.mode}")

    model = train(
        mode=args.mode,
        epochs=args.epochs,
        device=device,
        dtype=dtype,
    )
    evaluate(model, device, dtype)
    finite_difference_check(model, device, dtype)


if __name__ == "__main__":
    main()

