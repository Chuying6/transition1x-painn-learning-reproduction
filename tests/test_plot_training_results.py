from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from plot_training_results import (  # noqa: E402
    load_fixed_test_comparison,
    load_force_predictions,
    load_training_history,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def history_row(epoch: int, value: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "train_energy_rmse_eV": value,
        "train_force_rmse_eV_per_A": value,
        "val_energy_rmse_eV": value,
        "val_relative_energy_rmse_eV": value,
        "val_force_rmse_eV_per_A": value,
    }


def comparison_row(strategy: str, digest: str = "a" * 64) -> dict[str, object]:
    return {
        "training_strategy": strategy,
        "train_configurations": 1,
        "train_formulas": 1,
        "train_reactions": 1,
        "evaluation_hdf5": "fixed.h5",
        "evaluation_sha256": digest,
        "test_configurations": 150,
        "test_formulas": 8,
        "test_reactions": 15,
        "energy_rmse_eV": 1.0,
        "relative_energy_rmse_eV": 1.0,
        "force_rmse_eV_per_A": 1.0,
    }


def test_persisted_history_matches_json_and_best_epoch() -> None:
    epochs, series, best_epoch = load_training_history(
        PROJECT_ROOT / "results" / "expanded_training_history.csv",
        PROJECT_ROOT / "results" / "expanded_training_complete.json",
    )
    assert epochs.tolist() == list(range(1, 31))
    assert len(series["val_force_rmse_eV_per_A"]) == 30
    assert best_epoch == 23


def test_history_rejects_missing_epoch(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    write_csv(path, [history_row(1, 1.0), history_row(3, 0.5)])
    with pytest.raises(ValueError, match="contiguous"):
        load_training_history(path)


def test_history_rejects_json_drift(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    json_path = tmp_path / "history.json"
    rows = [history_row(1, 1.0), history_row(2, 0.5)]
    write_csv(csv_path, rows)
    changed_rows = [dict(row) for row in rows]
    changed_rows[1]["val_force_rmse_eV_per_A"] = 0.6
    json_path.write_text(
        json.dumps({"history": changed_rows, "best_epoch_by_val_force": 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match JSON"):
        load_training_history(csv_path, json_path)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "mini_diverse_equal_transition1x.h5").is_file(),
    reason="downloaded HDF5 files are intentionally excluded from Git",
)
def test_persisted_comparison_matches_local_fixed_test_file() -> None:
    rows = load_fixed_test_comparison(
        PROJECT_ROOT / "results" / "fixed_test_comparison.csv",
        PROJECT_ROOT / "data" / "mini_diverse_equal_transition1x.h5",
    )
    assert [row["training_strategy"] for row in rows] == [
        "Narrow",
        "Diverse equal-size",
        "Expanded diverse",
    ]


def test_comparison_rejects_different_test_hashes(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    rows = [
        comparison_row("Narrow"),
        comparison_row("Diverse equal-size"),
        comparison_row("Expanded diverse", digest="b" * 64),
    ]
    write_csv(path, rows)
    with pytest.raises(ValueError, match="same fixed test split"):
        load_fixed_test_comparison(path)


def test_comparison_rejects_missing_or_duplicate_strategy(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    rows = [
        comparison_row("Narrow"),
        comparison_row("Narrow"),
        comparison_row("Expanded diverse"),
    ]
    write_csv(path, rows)
    with pytest.raises(ValueError, match="each expected training strategy once"):
        load_fixed_test_comparison(path)


def test_force_predictions_load_reference_and_prediction_columns(tmp_path: Path) -> None:
    path = tmp_path / "forces.csv"
    write_csv(
        path,
        [
            {"reference_force_eV_per_A": -1.0, "predicted_force_eV_per_A": -0.8},
            {"reference_force_eV_per_A": 1.0, "predicted_force_eV_per_A": 0.9},
        ],
    )
    reference, predicted = load_force_predictions(path)
    assert reference.tolist() == [-1.0, 1.0]
    assert predicted.tolist() == [-0.8, 0.9]


def test_force_predictions_reject_constant_references(tmp_path: Path) -> None:
    path = tmp_path / "forces.csv"
    write_csv(
        path,
        [
            {"reference_force_eV_per_A": 1.0, "predicted_force_eV_per_A": 0.8},
            {"reference_force_eV_per_A": 1.0, "predicted_force_eV_per_A": 0.9},
        ],
    )
    with pytest.raises(ValueError, match="distinct references"):
        load_force_predictions(path)
