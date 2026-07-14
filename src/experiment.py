"""Experiment tracking and registry for WAXAL ASR.

Manages the full lifecycle of a training experiment:
- Creates a timestamped experiment directory under ``experiments/``
- Saves a frozen copy of the config, git hash, and environment info
- Tracks the best checkpoint by validation WER
- Appends a summary row to ``experiments/experiment_registry.csv``
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

_REGISTRY_COLUMNS = [
    "timestamp",
    "experiment_name",
    "model",
    "languages",
    "learning_rate",
    "batch_size",
    "max_steps",
    "lora_r",
    "lora_alpha",
    "lora_targets",
    "wer",
    "cer",
    "best_step",
    "checkpoint_path",
    "git_commit",
    "notes",
]

_EXPERIMENTS_ROOT = Path("experiments")
_REGISTRY_PATH = _EXPERIMENTS_ROOT / "experiment_registry.csv"


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def get_git_commit_hash() -> str:
    """Return the current short git commit hash, or ``'unknown'`` if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def get_git_diff_status() -> str:
    """Return ``'clean'`` or ``'dirty'`` depending on uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            timeout=5,
        )
        return "clean" if result.returncode == 0 else "dirty"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

def collect_environment_info() -> dict[str, str]:
    """Collect key software and hardware versions into a dict."""
    info: dict[str, str] = {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = str(torch.version.cuda)
        info["cudnn_version"] = str(torch.backends.cudnn.version())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f"gpu_{i}"] = f"{props.name} ({props.total_mem / (1024**3):.1f} GB)"

    for pkg_name in ("transformers", "datasets", "peft", "trl", "jiwer", "numpy", "librosa"):
        try:
            mod = __import__(pkg_name)
            info[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg_name] = "not installed"

    info["git_commit"] = get_git_commit_hash()
    info["git_status"] = get_git_diff_status()
    return info


# ---------------------------------------------------------------------------
# Experiment directory management
# ---------------------------------------------------------------------------

class Experiment:
    """Manages the directory and artifacts for a single experiment run.

    Usage::

        exp = Experiment.create(config)
        # ... training happens ...
        exp.log_metrics({"wer": 0.52, "cer": 0.14})
        exp.save_predictions(refs, preds)
        exp.register(metrics)
    """

    def __init__(self, experiment_dir: Path, config: dict[str, Any]) -> None:
        self.dir = experiment_dir
        self.config = config
        self.name = experiment_dir.name
        self.start_time = datetime.now(timezone.utc)
        self._best_wer: float = float("inf")
        self._best_step: int = -1
        self._best_checkpoint_path: str = ""
        self._training_log: list[dict[str, Any]] = []

    # ---- Factory ----

    @classmethod
    def create(
        cls,
        config: dict[str, Any],
        name: str | None = None,
        notes: str = "",
    ) -> Experiment:
        """Create a new experiment directory with all metadata saved.

        Args:
            config: The full merged configuration dict.
            name: Optional human-readable name.  Defaults to a timestamp.
            notes: Free-text notes for the registry.

        Returns:
            An :class:`Experiment` instance.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        languages = "-".join(config["dataset"]["languages"])
        default_name = f"{timestamp}_{languages}"
        exp_name = name or default_name

        exp_dir = _EXPERIMENTS_ROOT / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "predictions").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)

        exp = cls(exp_dir, config)
        exp._notes = notes

        # Save frozen config
        with open(exp_dir / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

        # Save environment snapshot
        env_info = collect_environment_info()
        with open(exp_dir / "environment.json", "w", encoding="utf-8") as fh:
            json.dump(env_info, fh, indent=2)

        logger.info("Experiment created: %s", exp_dir)
        return exp

    # ---- Training log ----

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        """Append a training step's metrics to the in-memory log.

        Args:
            step: The global training step number.
            metrics: Dict of metric values (loss, eval_loss, etc.).
        """
        entry = {"step": step, "timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
        self._training_log.append(entry)

    def save_training_log(self) -> Path:
        """Flush the accumulated training log to a JSON file.

        Returns:
            Path to the saved log file.
        """
        log_path = self.dir / "logs" / "training_log.json"
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(self._training_log, fh, indent=2)
        logger.info("Training log saved (%d entries) to %s", len(self._training_log), log_path)
        return log_path

    # ---- Best checkpoint tracking ----

    def update_best(self, step: int, wer: float, checkpoint_path: str) -> bool:
        """Update the best checkpoint if *wer* improves.

        Args:
            step: Training step.
            wer: Validation WER at this step.
            checkpoint_path: Path to the checkpoint directory.

        Returns:
            ``True`` if this is a new best.
        """
        if wer < self._best_wer:
            self._best_wer = wer
            self._best_step = step
            self._best_checkpoint_path = checkpoint_path
            logger.info("New best WER=%.4f at step %d (checkpoint: %s)", wer, step, checkpoint_path)
            return True
        return False

    @property
    def best_wer(self) -> float:
        """The best validation WER observed so far."""
        return self._best_wer

    @property
    def best_step(self) -> int:
        """The step at which the best WER was achieved."""
        return self._best_step

    # ---- Metrics persistence ----

    def log_metrics(self, metrics: dict[str, float], tag: str = "eval") -> Path:
        """Save evaluation metrics to a JSON file inside the experiment.

        Args:
            metrics: Dict of metric name to value.
            tag: Sub-label (e.g. ``"eval"``, ``"test"``).

        Returns:
            Path to the saved metrics file.
        """
        out = self.dir / f"metrics_{tag}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Metrics [%s] saved to %s", tag, out)
        return out

    def save_predictions(
        self,
        references: list[str],
        predictions: list[str],
        tag: str = "eval",
    ) -> Path:
        """Save reference/prediction pairs to JSON.

        Args:
            references: Ground-truth transcriptions.
            predictions: Model predictions.
            tag: Sub-label for the file.

        Returns:
            Path to the saved predictions file.
        """
        out = self.dir / "predictions" / f"predictions_{tag}.json"
        entries = []
        for i, (ref, pred) in enumerate(zip(references, predictions)):
            entries.append({"index": i, "reference": ref, "prediction": pred})
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d predictions to %s", len(entries), out)
        return out

    # ---- Experiment registry ----

    def register(self, metrics: dict[str, float] | None = None) -> None:
        """Append a summary row to the global experiment registry CSV.

        Args:
            metrics: Final evaluation metrics (``wer``, ``cer``).
        """
        _EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
        file_exists = _REGISTRY_PATH.exists()

        metrics = metrics or {}
        lora_cfg = self.config.get("lora", {})
        training_cfg = self.config.get("training", {})
        dataset_cfg = self.config.get("dataset", {})
        model_cfg = self.config.get("model", {})

        row = {
            "timestamp": self.start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "experiment_name": self.name,
            "model": model_cfg.get("model_id", ""),
            "languages": "-".join(dataset_cfg.get("languages", [])),
            "learning_rate": training_cfg.get("learning_rate", ""),
            "batch_size": training_cfg.get("per_device_train_batch_size", ""),
            "max_steps": training_cfg.get("max_steps", ""),
            "lora_r": lora_cfg.get("r", ""),
            "lora_alpha": lora_cfg.get("alpha", ""),
            "lora_targets": ",".join(lora_cfg.get("target_modules", [])),
            "wer": f"{metrics.get('wer', ''):.4f}" if isinstance(metrics.get("wer"), float) else "",
            "cer": f"{metrics.get('cer', ''):.4f}" if isinstance(metrics.get("cer"), float) else "",
            "best_step": self._best_step if self._best_step >= 0 else "",
            "checkpoint_path": self._best_checkpoint_path,
            "git_commit": get_git_commit_hash(),
            "notes": getattr(self, "_notes", ""),
        }

        with open(_REGISTRY_PATH, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_REGISTRY_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info("Experiment registered in %s", _REGISTRY_PATH)
