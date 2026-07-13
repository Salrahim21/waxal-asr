"""Utility helpers for WAXAL ASR.

Covers reproducibility, GPU diagnostics, memory reporting, and logging setup.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent format.

    Args:
        level: Logging level (default ``logging.INFO``).
    """
    fmt = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# GPU / System info
# ---------------------------------------------------------------------------

def log_gpu_info() -> None:
    """Log GPU name, memory, and CUDA version if available."""
    if not torch.cuda.is_available():
        logger.info("No CUDA GPU detected — running on CPU.")
        return

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_mem / (1024 ** 3)
        logger.info(
            "GPU %d: %s — %.1f GB VRAM — CUDA %d.%d",
            i,
            props.name,
            total_gb,
            props.major,
            props.minor,
        )
    logger.info("PyTorch CUDA version: %s", torch.version.cuda)
    logger.info("cuDNN version: %s", torch.backends.cudnn.version())


def log_memory_usage() -> None:
    """Log current GPU memory allocation (allocated / reserved)."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        logger.info(
            "GPU %d memory — allocated: %.2f GB, reserved: %.2f GB",
            i,
            alloc,
            reserved,
        )


def log_environment() -> None:
    """Log key software versions for reproducibility records."""
    import transformers
    import datasets
    import peft
    import trl

    logger.info("Python       : %s", sys.version.split()[0])
    logger.info("PyTorch      : %s", torch.__version__)
    logger.info("Transformers : %s", transformers.__version__)
    logger.info("Datasets     : %s", datasets.__version__)
    logger.info("PEFT         : %s", peft.__version__)
    logger.info("TRL          : %s", trl.__version__)
    logger.info("NumPy        : %s", np.__version__)


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
