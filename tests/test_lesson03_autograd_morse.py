from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lesson03_autograd_morse import (  # noqa: E402
    EnergyNetwork,
    MorseParameters,
    energy_and_force,
    make_dataset,
    morse_energy,
    morse_force,
)


class MorseAutogradTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float64

    def test_analytical_force_is_negative_energy_gradient(self) -> None:
        q = torch.tensor(
            [[-0.20], [0.00], [0.40], [1.20]],
            dtype=self.dtype,
            requires_grad=True,
        )
        params = MorseParameters()
        energy = morse_energy(q, params)
        force_from_gradient = -torch.autograd.grad(energy.sum(), q)[0]
        torch.testing.assert_close(
            morse_force(q, params), force_from_gradient, rtol=1.0e-12, atol=1.0e-12
        )

    def test_model_force_matches_central_finite_difference(self) -> None:
        torch.manual_seed(7)
        model = EnergyNetwork(hidden_features=8).to(dtype=self.dtype)
        q_value = 0.40
        step = 1.0e-5
        q = torch.tensor([[q_value]], dtype=self.dtype, requires_grad=True)
        _, force_autograd = energy_and_force(model, q, create_graph=False)
        with torch.no_grad():
            q_plus = torch.tensor([[q_value + step]], dtype=self.dtype)
            q_minus = torch.tensor([[q_value - step]], dtype=self.dtype)
            force_finite_difference = -(model(q_plus) - model(q_minus)) / (2.0 * step)
        torch.testing.assert_close(
            force_autograd,
            force_finite_difference,
            rtol=1.0e-8,
            atol=1.0e-8,
        )

    def test_force_loss_backpropagates_to_model_parameters(self) -> None:
        torch.manual_seed(7)
        model = EnergyNetwork(hidden_features=8).to(dtype=self.dtype)
        q = torch.tensor([[-0.10], [0.20], [0.50]], dtype=self.dtype, requires_grad=True)
        _, predicted_force = energy_and_force(model, q, create_graph=True)
        predicted_force.square().mean().backward()
        parameters = dict(model.named_parameters())
        shape_gradients = [
            parameter.grad
            for name, parameter in parameters.items()
            if name != "layers.4.bias"
        ]
        self.assertTrue(all(gradient is not None for gradient in shape_gradients))
        self.assertTrue(
            any(torch.count_nonzero(gradient).item() for gradient in shape_gradients)
        )

        # A constant energy offset disappears under -dE/dq, so force-only loss
        # cannot determine the final scalar bias (the energy zero).
        self.assertIsNone(parameters["layers.4.bias"].grad)

    def test_training_and_test_ranges_do_not_overlap(self) -> None:
        q_train, _, _, q_test, _, _ = make_dataset(self.device, self.dtype)
        self.assertLess(q_train.max().item(), q_test.min().item())


if __name__ == "__main__":
    unittest.main()
