"""Rich console logging for WAXAL ASR.

Provides colored log output, GPU memory bars, dataset statistics summaries,
epoch timing with ETA, and best-checkpoint banners.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI colour codes (work on most modern terminals including Windows 10+)
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_MAGENTA = "\033[95m"
_CYAN = "\033[96m"

_LEVEL_COLORS = {
    "DEBUG": _DIM,
    "INFO": _CYAN,
    "WARNING": _YELLOW,
    "ERROR": _RED,
    "CRITICAL": f"{_BOLD}{_RED}",
}


class ColoredFormatter(logging.Formatter):
    """Log formatter that adds ANSI colours to the level name and timestamp."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        record.name = f"{_DIM}{record.name}{_RESET}"
        record.asctime = f"{_BLUE}{self.formatTime(record, self.datefmt)}{_RESET}"
        return super().format(record)


def setup_colored_logging(level: int = logging.INFO) -> None:
    """Replace the root logger handler with a coloured one.

    Args:
        level: Logging level.
    """
    fmt = "%(asctime)s %(levelname)s %(name)s — %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# GPU memory bar
# ---------------------------------------------------------------------------

def gpu_memory_bar(device_idx: int = 0, bar_width: int = 30) -> str:
    """Return a text progress bar showing GPU memory usage.

    Args:
        device_idx: CUDA device index.
        bar_width: Character width of the bar.

    Returns:
        A formatted string like ``[======>         ] 4.2 / 15.0 GB (28%)``.
    """
    if not torch.cuda.is_available():
        return "No GPU available"

    alloc = torch.cuda.memory_allocated(device_idx)
    total = torch.cuda.get_device_properties(device_idx).total_mem
    ratio = alloc / total if total > 0 else 0
    filled = int(bar_width * ratio)
    bar = "=" * filled + ">" + " " * (bar_width - filled - 1)
    alloc_gb = alloc / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    pct = ratio * 100
    return f"[{bar}] {alloc_gb:.1f} / {total_gb:.1f} GB ({pct:.0f}%)"


def log_gpu_memory_bar() -> None:
    """Log a GPU memory bar for each available device."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        logger.info("GPU %d memory: %s", i, gpu_memory_bar(i))


# ---------------------------------------------------------------------------
# Dataset statistics summary
# ---------------------------------------------------------------------------

def log_dataset_summary(
    num_train: int,
    num_val: int,
    languages: list[str],
    audio_stats: dict[str, float] | None = None,
) -> None:
    """Log a formatted dataset summary block.

    Args:
        num_train: Number of training examples.
        num_val: Number of validation examples.
        languages: List of language codes.
        audio_stats: Optional dict with ``min_duration``, ``max_duration``,
            ``mean_duration`` in seconds.
    """
    logger.info(f"{_BOLD}{_GREEN}{'=' * 50}{_RESET}")
    logger.info(f"{_BOLD}{_GREEN}  Dataset Summary{_RESET}")
    logger.info(f"{_BOLD}{_GREEN}{'=' * 50}{_RESET}")
    logger.info("  Languages      : %s", ", ".join(languages))
    logger.info("  Train examples : %d", num_train)
    logger.info("  Val examples   : %d", num_val)
    if audio_stats:
        logger.info(
            "  Audio duration : %.1fs min / %.1fs max / %.1fs mean",
            audio_stats.get("min_duration", 0),
            audio_stats.get("max_duration", 0),
            audio_stats.get("mean_duration", 0),
        )
    logger.info(f"{_BOLD}{_GREEN}{'=' * 50}{_RESET}")


# ---------------------------------------------------------------------------
# Training timer with ETA
# ---------------------------------------------------------------------------

class TrainingTimer:
    """Tracks elapsed time and estimates remaining time for training.

    Usage::

        timer = TrainingTimer(total_steps=3000)
        for step in range(3000):
            # ... train ...
            timer.step()
            if step % 100 == 0:
                timer.log_progress(step)
    """

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self._start_time = time.monotonic()
        self._step_count = 0

    def step(self) -> None:
        """Record one completed step."""
        self._step_count += 1

    def elapsed(self) -> float:
        """Return elapsed time in seconds since construction."""
        return time.monotonic() - self._start_time

    def eta_seconds(self) -> float:
        """Estimate remaining seconds based on average step time."""
        if self._step_count == 0:
            return 0.0
        avg = self.elapsed() / self._step_count
        remaining = self.total_steps - self._step_count
        return max(0.0, avg * remaining)

    def log_progress(self, current_step: int, extra: str = "") -> None:
        """Log a progress line with elapsed time and ETA.

        Args:
            current_step: Current global step.
            extra: Additional text to append (e.g. loss info).
        """
        elapsed = self.elapsed()
        eta = self.eta_seconds()
        pct = (current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        elapsed_str = _format_duration(elapsed)
        eta_str = _format_duration(eta)
        msg = (
            f"Step {current_step}/{self.total_steps} ({pct:.1f}%) "
            f"| Elapsed: {elapsed_str} | ETA: {eta_str}"
        )
        if extra:
            msg += f" | {extra}"
        logger.info(msg)


# ---------------------------------------------------------------------------
# Best checkpoint banner
# ---------------------------------------------------------------------------

def log_best_checkpoint(step: int, wer: float, cer: float, path: str) -> None:
    """Log a highlighted banner when a new best checkpoint is found.

    Args:
        step: Training step.
        wer: Validation WER.
        cer: Validation CER.
        path: Checkpoint path.
    """
    logger.info(f"{_BOLD}{_MAGENTA}{'*' * 50}{_RESET}")
    logger.info(f"{_BOLD}{_MAGENTA}  NEW BEST CHECKPOINT{_RESET}")
    logger.info(f"{_BOLD}{_MAGENTA}  Step : %d{_RESET}", step)
    logger.info(f"{_BOLD}{_MAGENTA}  WER  : %.2f%%{_RESET}", wer * 100)
    logger.info(f"{_BOLD}{_MAGENTA}  CER  : %.2f%%{_RESET}", cer * 100)
    logger.info(f"{_BOLD}{_MAGENTA}  Path : %s{_RESET}", path)
    logger.info(f"{_BOLD}{_MAGENTA}{'*' * 50}{_RESET}")


# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

def log_training_summary(
    total_steps: int,
    elapsed_seconds: float,
    final_loss: float | None,
    best_wer: float,
    best_step: int,
) -> None:
    """Log a final training summary block.

    Args:
        total_steps: Total steps completed.
        elapsed_seconds: Total training wall time.
        final_loss: Final training loss (or ``None``).
        best_wer: Best validation WER achieved.
        best_step: Step at which best WER was achieved.
    """
    logger.info(f"{_BOLD}{_CYAN}{'=' * 50}{_RESET}")
    logger.info(f"{_BOLD}{_CYAN}  Training Complete{_RESET}")
    logger.info(f"{_BOLD}{_CYAN}{'=' * 50}{_RESET}")
    logger.info("  Total steps  : %d", total_steps)
    logger.info("  Wall time    : %s", _format_duration(elapsed_seconds))
    if final_loss is not None:
        logger.info("  Final loss   : %.4f", final_loss)
    if best_wer < float("inf"):
        logger.info("  Best WER     : %.2f%% (step %d)", best_wer * 100, best_step)
    logger.info(f"{_BOLD}{_CYAN}{'=' * 50}{_RESET}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format seconds into ``HH:MM:SS``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
