"""Visualization utilities for WAXAL ASR.

Generates Matplotlib plots for training monitoring and experiment analysis:
- Training loss curves
- Validation WER/CER curves
- Audio duration distributions
- Transcript length distributions
- Prediction comparison tables
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/notebook use
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed — visualization disabled.")


def _check_matplotlib() -> None:
    """Raise if matplotlib is not available."""
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


# ---------------------------------------------------------------------------
# Training loss curve
# ---------------------------------------------------------------------------

def plot_training_loss(
    training_log: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Training Loss",
) -> Path:
    """Plot training loss over steps.

    Args:
        training_log: List of dicts with ``step`` and ``loss`` keys.
        output_path: Path to save the PNG plot.
        title: Plot title.

    Returns:
        Path to the saved plot.
    """
    _check_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = [e["step"] for e in training_log if "loss" in e]
    losses = [e["loss"] for e in training_log if "loss" in e]

    if not steps:
        logger.warning("No loss entries in training log — skipping plot.")
        return output_path

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color="#1f77b4", linewidth=1.5, alpha=0.8)

    # Smoothed line (exponential moving average)
    if len(losses) > 10:
        alpha_ema = 0.1
        smoothed = [losses[0]]
        for val in losses[1:]:
            smoothed.append(alpha_ema * val + (1 - alpha_ema) * smoothed[-1])
        ax.plot(steps, smoothed, color="#ff7f0e", linewidth=2, label="EMA (smoothed)")
        ax.legend()

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Training loss plot saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Validation WER / CER curves
# ---------------------------------------------------------------------------

def plot_eval_metrics(
    training_log: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Validation Metrics",
) -> Path:
    """Plot WER and CER curves over training steps.

    Args:
        training_log: List of dicts, where eval steps include ``eval_wer``
            and ``eval_cer`` keys.
        output_path: Path to save the PNG plot.
        title: Plot title.

    Returns:
        Path to the saved plot.
    """
    _check_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eval_entries = [e for e in training_log if "eval_wer" in e]
    if not eval_entries:
        logger.warning("No eval metrics in training log — skipping plot.")
        return output_path

    steps = [e["step"] for e in eval_entries]
    wers = [e["eval_wer"] * 100 for e in eval_entries]
    cers = [e["eval_cer"] * 100 for e in eval_entries]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_wer = "#d62728"
    color_cer = "#2ca02c"

    ax1.plot(steps, wers, color=color_wer, linewidth=2, marker="o", markersize=5, label="WER")
    ax1.plot(steps, cers, color=color_cer, linewidth=2, marker="s", markersize=5, label="CER")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Error Rate (%)")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))

    # Mark best WER
    if wers:
        best_idx = int(np.argmin(wers))
        ax1.annotate(
            f"Best WER: {wers[best_idx]:.1f}%",
            xy=(steps[best_idx], wers[best_idx]),
            xytext=(10, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color_wer),
            fontsize=9,
            color=color_wer,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Eval metrics plot saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Audio duration distribution
# ---------------------------------------------------------------------------

def plot_audio_duration_distribution(
    durations: list[float],
    output_path: str | Path,
    title: str = "Audio Duration Distribution",
) -> Path:
    """Plot histogram of audio durations in seconds.

    Args:
        durations: List of audio durations in seconds.
        output_path: Path to save the PNG plot.
        title: Plot title.

    Returns:
        Path to the saved plot.
    """
    _check_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(durations, bins=50, color="#1f77b4", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(durations), color="#d62728", linestyle="--", label=f"Mean: {np.mean(durations):.1f}s")
    ax.axvline(np.median(durations), color="#ff7f0e", linestyle="--", label=f"Median: {np.median(durations):.1f}s")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Audio duration plot saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Transcript length distribution
# ---------------------------------------------------------------------------

def plot_transcript_length_distribution(
    lengths: list[int],
    output_path: str | Path,
    title: str = "Transcript Length Distribution",
    unit: str = "characters",
) -> Path:
    """Plot histogram of transcript lengths.

    Args:
        lengths: List of transcript lengths (in chars or words).
        output_path: Path to save the PNG plot.
        title: Plot title.
        unit: Label for x-axis (``"characters"`` or ``"words"``).

    Returns:
        Path to the saved plot.
    """
    _check_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=50, color="#2ca02c", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(lengths), color="#d62728", linestyle="--", label=f"Mean: {np.mean(lengths):.0f}")
    ax.set_xlabel(f"Length ({unit})")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Transcript length plot saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Prediction comparison table
# ---------------------------------------------------------------------------

def plot_prediction_table(
    references: list[str],
    predictions: list[str],
    wers: list[float],
    output_path: str | Path,
    num_rows: int = 15,
    title: str = "Prediction Comparison",
) -> Path:
    """Create a table image comparing references to predictions.

    Shows the *num_rows* examples with highest WER first (worst predictions).

    Args:
        references: Reference transcriptions.
        predictions: Model predictions.
        wers: Per-example WER values.
        output_path: Path to save the PNG.
        num_rows: Number of rows to display.
        title: Table title.

    Returns:
        Path to the saved plot.
    """
    _check_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by WER descending (worst first)
    indices = np.argsort(wers)[::-1][:num_rows]
    max_chars = 60

    def _trunc(s: str) -> str:
        return s[:max_chars] + "..." if len(s) > max_chars else s

    cell_text = []
    for idx in indices:
        cell_text.append([
            str(idx),
            _trunc(references[idx]),
            _trunc(predictions[idx]),
            f"{wers[idx]:.2%}",
        ])

    fig, ax = plt.subplots(figsize=(16, max(4, num_rows * 0.5 + 1)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    table = ax.table(
        cellText=cell_text,
        colLabels=["#", "Reference", "Prediction", "WER"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width([0, 1, 2, 3])

    # Style header row
    for j in range(4):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(1, len(cell_text) + 1):
        bg = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(4):
            table[i, j].set_facecolor(bg)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Prediction comparison table saved to %s", output_path)
    return output_path
