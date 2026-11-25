#!/usr/bin/env python3
"""Render publication-style figures for the mini Transition1x reproduction.

Figure 1 asks whether the expanded-data PaiNN training run converged stably.
Figure 2 asks whether increasing reaction coverage improved generalization on
one fixed held-out split. Figure 3 compares predicted and reference test-force
components and exposes force-dependent residuals. The figures inherit only the
visual grammar of the figures4papers examples; all code and plotted values are
project-specific.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

try:  # Optional local QA helper; it is not required to reproduce the figures.
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:  # pragma: no cover - depends on the local authoring environment
    require_matplotlib_panel_alignment = None


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#8BCF8B",
    "red": "#B64342",
    "neutral": "#767676",
    "black": "#272727",
}

HISTORY_COLUMNS = (
    "train_energy_rmse_eV",
    "train_force_rmse_eV_per_A",
    "val_energy_rmse_eV",
    "val_relative_energy_rmse_eV",
    "val_force_rmse_eV_per_A",
)
COMPARISON_METRICS = (
    "energy_rmse_eV",
    "relative_energy_rmse_eV",
    "force_rmse_eV_per_A",
)
STRATEGY_LABELS = {
    "Narrow": "Narrow",
    "Diverse equal-size": "Diverse",
    "Expanded diverse": "Expanded",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "lines.linewidth": 1.7,
            "savefig.facecolor": "white",
        }
    )


def read_numeric_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _positive_finite(row: dict[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{column} must contain finite positive values")
    return value


def load_training_history(
    history_csv: Path, history_json: Path | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Load and validate a complete epoch history and its optional source JSON."""

    rows = read_numeric_csv(history_csv)
    try:
        epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=int)
    except (KeyError, ValueError) as exc:
        raise ValueError("epoch must contain integers") from exc
    expected = np.arange(epochs[0], epochs[-1] + 1)
    if epochs[0] < 1 or not np.array_equal(epochs, expected):
        raise ValueError("epochs must be positive, strictly increasing, and contiguous")

    series = {
        column: np.asarray([_positive_finite(row, column) for row in rows])
        for column in HISTORY_COLUMNS
    }
    best_epoch = int(epochs[np.argmin(series["val_force_rmse_eV_per_A"])])

    if history_json is not None:
        payload: dict[str, Any] = json.loads(history_json.read_text(encoding="utf-8"))
        source_rows = payload.get("history")
        if not isinstance(source_rows, list) or len(source_rows) != len(rows):
            raise ValueError("history CSV and JSON contain different row counts")
        source_epochs = [source.get("epoch") for source in source_rows]
        if source_epochs != epochs.tolist():
            raise ValueError("history CSV and JSON contain different epochs")
        for column in HISTORY_COLUMNS:
            source_values = np.asarray([float(source[column]) for source in source_rows])
            if not np.allclose(series[column], source_values, rtol=0.0, atol=5.1e-7):
                raise ValueError(f"history CSV does not match JSON for {column}")
        if payload.get("best_epoch_by_val_force") != best_epoch:
            raise ValueError("recorded best epoch does not match validation force history")
    return epochs, series, best_epoch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_test_comparison(
    comparison_csv: Path, evaluation_hdf5: Path | None = None
) -> list[dict[str, str]]:
    """Validate metrics and fixed-test provenance before plotting."""

    rows = read_numeric_csv(comparison_csv)
    strategies = [row.get("training_strategy") for row in rows]
    if len(rows) != len(STRATEGY_LABELS) or set(strategies) != set(STRATEGY_LABELS):
        raise ValueError("comparison must contain each expected training strategy once")
    test_signatures = {
        (
            row.get("evaluation_hdf5"),
            row.get("evaluation_sha256"),
            row.get("test_configurations"),
            row.get("test_formulas"),
            row.get("test_reactions"),
        )
        for row in rows
    }
    if len(test_signatures) != 1:
        raise ValueError("all strategies must use the same fixed test split")
    signature = next(iter(test_signatures))
    if not signature[0] or not signature[1]:
        raise ValueError("fixed-test filename and SHA-256 are required")
    for row in rows:
        for column in COMPARISON_METRICS:
            _positive_finite(row, column)
    if evaluation_hdf5 is not None:
        if not evaluation_hdf5.is_file():
            raise FileNotFoundError(f"evaluation HDF5 does not exist: {evaluation_hdf5}")
        if evaluation_hdf5.name != signature[0]:
            raise ValueError("evaluation HDF5 filename does not match comparison CSV")
        if sha256_file(evaluation_hdf5) != signature[1]:
            raise ValueError("evaluation HDF5 SHA-256 does not match comparison CSV")
    return rows


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-13, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def save_figure(fig: plt.Figure, basename: Path, panel_ids: list[str]) -> None:
    basename.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0, w_pad=1.6)
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            axes=list(fig.axes),
            panel_ids=panel_ids,
            row_groups=[panel_ids],
            json_out=str(basename) + ".alignment.json",
            overlay_svg=str(basename) + ".alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
    fig.savefig(str(basename) + ".svg", bbox_inches="tight")
    fig.savefig(str(basename) + ".pdf", bbox_inches="tight")
    fig.savefig(str(basename) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(basename) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history_csv: Path, output_dir: Path, history_json: Path | None = None
) -> None:
    epochs, series, best_epoch = load_training_history(history_csv, history_json)
    marker_indices = list(range(0, len(epochs), 3))
    if marker_indices[-1] != len(epochs) - 1:
        marker_indices.append(len(epochs) - 1)
    val_force = series["val_force_rmse_eV_per_A"]
    best_index = int(np.argmin(val_force))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))

    axes[0].plot(
        epochs,
        series["train_energy_rmse_eV"],
        color=PALETTE["blue_main"],
        marker="o",
        markersize=2.5,
        markevery=marker_indices,
        label="Train absolute",
    )
    axes[0].plot(
        epochs,
        series["val_energy_rmse_eV"],
        color=PALETTE["red"],
        marker="s",
        markersize=2.5,
        markevery=marker_indices,
        label="Validation absolute",
    )
    axes[0].plot(
        epochs,
        series["val_relative_energy_rmse_eV"],
        color=PALETTE["green"],
        marker="^",
        markersize=2.7,
        markevery=marker_indices,
        label="Validation reaction-relative",
    )
    axes[0].set(xlabel="Epoch", ylabel="Energy RMSE (eV)", title="Energy learning")
    axes[0].legend(loc="upper right")
    add_panel_label(axes[0], "a")

    axes[1].plot(
        epochs,
        series["train_force_rmse_eV_per_A"],
        color=PALETTE["blue_main"],
        marker="o",
        markersize=2.5,
        markevery=marker_indices,
        label="Train",
    )
    axes[1].plot(
        epochs,
        val_force,
        color=PALETTE["red"],
        marker="s",
        markersize=2.5,
        markevery=marker_indices,
        label="Validation",
    )
    axes[1].scatter(
        [best_epoch],
        [val_force[best_index]],
        marker="*",
        s=55,
        color="#FFD700",
        edgecolor=PALETTE["black"],
        linewidth=0.7,
        zorder=5,
        label=f"Best checkpoint (epoch {best_epoch})",
    )
    axes[1].set(
        xlabel="Epoch", ylabel="Force RMSE (eV/Å)", title="Force learning"
    )
    axes[1].legend(loc="upper right")
    add_panel_label(axes[1], "b")

    for ax in axes:
        ax.set_xlim(int(epochs[0]), int(epochs[-1]))
        ax.set_xticks(np.unique(np.linspace(epochs[0], epochs[-1], 7, dtype=int)))
        ax.tick_params(direction="out", length=3, width=0.8)

    save_figure(fig, output_dir / "expanded_training_curves", ["a", "b"])


def plot_fixed_test_comparison(
    comparison_csv: Path,
    output_dir: Path,
    evaluation_hdf5: Path | None = None,
) -> None:
    rows = load_fixed_test_comparison(comparison_csv, evaluation_hdf5)
    labels = [STRATEGY_LABELS[row["training_strategy"]] for row in rows]
    colors = [PALETTE["red"], PALETTE["green"], PALETTE["blue_main"]]
    hatches = ["//", "..", ""]
    metrics = [
        ("energy_rmse_eV", "Absolute energy", "RMSE (eV)"),
        ("relative_energy_rmse_eV", "Reaction-relative energy", "RMSE (eV)"),
        ("force_rmse_eV_per_A", "Forces", "RMSE (eV/Å)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85))
    x = np.arange(len(rows))
    for panel, (ax, (column, title, ylabel)) in enumerate(zip(axes, metrics)):
        values = np.asarray([float(row[column]) for row in rows])
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{column} must contain finite positive values")
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor=PALETTE["black"],
            linewidth=0.8,
            width=0.68,
        )
        for bar, hatch, value in zip(bars, hatches, values):
            bar.set_hatch(hatch)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.09,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.1,
            )
        ax.set_yscale("log")
        ax.set_ylim(values.min() / 1.45, values.max() * 1.55)
        tick_candidates = np.asarray([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        visible_ticks = tick_candidates[
            (tick_candidates >= ax.get_ylim()[0])
            & (tick_candidates <= ax.get_ylim()[1])
        ]
        ax.yaxis.set_major_locator(FixedLocator(visible_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.minorticks_off()
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(direction="out", length=3, width=0.8)
        add_panel_label(ax, chr(ord("a") + panel))

    save_figure(fig, output_dir / "fixed_test_comparison", ["a", "b", "c"])


def load_force_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_numeric_csv(path)
    try:
        reference = np.asarray(
            [float(row["reference_force_eV_per_A"]) for row in rows]
        )
        predicted = np.asarray(
            [float(row["predicted_force_eV_per_A"]) for row in rows]
        )
    except (KeyError, ValueError) as exc:
        raise ValueError("force prediction CSV contains invalid numeric data") from exc
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(predicted)):
        raise ValueError("force predictions must be finite")
    if len(reference) < 2 or np.ptp(reference) == 0.0:
        raise ValueError("force predictions require at least two distinct references")
    return reference, predicted


def plot_force_parity(force_csv: Path, output_dir: Path) -> None:
    reference, predicted = load_force_predictions(force_csv)
    residual = predicted - reference
    rmse = float(np.sqrt(np.mean(residual**2)))
    r_squared = float(
        1.0 - np.sum(residual**2) / np.sum((reference - np.mean(reference)) ** 2)
    )
    limit = float(max(np.max(np.abs(reference)), np.max(np.abs(predicted))))
    padding = max(0.05 * limit, 0.02)
    bounds = (-limit - padding, limit + padding)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    axes[0].hexbin(
        reference,
        predicted,
        gridsize=70,
        mincnt=1,
        bins="log",
        cmap="Blues",
        linewidths=0.0,
    )
    axes[0].plot(bounds, bounds, color=PALETTE["red"], linestyle="--", linewidth=1.2)
    axes[0].set(
        xlim=bounds,
        ylim=bounds,
        aspect="equal",
        xlabel="Reference force component (eV/Å)",
        ylabel="Predicted force component (eV/Å)",
        title="Test-force parity",
    )
    axes[0].text(
        0.04,
        0.96,
        f"RMSE = {rmse:.3f} eV/Å\n$R^2$ = {r_squared:.3f}\n$n$ = {len(reference):,}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
    )
    add_panel_label(axes[0], "a")

    axes[1].hexbin(
        reference,
        residual,
        gridsize=70,
        mincnt=1,
        bins="log",
        cmap="Blues",
        linewidths=0.0,
    )
    axes[1].axhline(0.0, color=PALETTE["red"], linestyle="--", linewidth=1.2)
    residual_limit = float(np.max(np.abs(residual)))
    residual_padding = max(0.05 * residual_limit, 0.02)
    axes[1].set(
        xlim=bounds,
        ylim=(-residual_limit - residual_padding, residual_limit + residual_padding),
        xlabel="Reference force component (eV/Å)",
        ylabel="Prediction residual (eV/Å)",
        title="Force residuals",
    )
    add_panel_label(axes[1], "b")

    for ax in axes:
        ax.tick_params(direction="out", length=3, width=0.8)
    save_figure(fig, output_dir / "expanded_test_force_parity", ["a", "b"])


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=project_root / "results" / "expanded_training_history.csv",
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        default=project_root / "results" / "expanded_training_complete.json",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=project_root / "results" / "fixed_test_comparison.csv",
    )
    parser.add_argument(
        "--evaluation-hdf5",
        type=Path,
        default=None,
        help="optional local HDF5 file whose SHA-256 is checked against the CSV",
    )
    parser.add_argument(
        "--force-predictions",
        type=Path,
        default=project_root / "results" / "expanded_test_force_predictions.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "figures"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_publication_style()
    plot_training_curves(args.history, args.output_dir, args.history_json)
    plot_fixed_test_comparison(args.comparison, args.output_dir, args.evaluation_hdf5)
    plot_force_parity(args.force_predictions, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
