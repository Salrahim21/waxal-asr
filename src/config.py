"""Configuration loading and merging for WAXAL ASR.

Loads a base YAML config and optionally deep-merges an override YAML on top.
All downstream modules receive a plain ``dict`` so there is no coupling to a
specific config library.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*.

    Scalar values in *override* replace those in *base*; nested dicts are
    merged recursively; lists are replaced wholesale.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    override_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the default config, optionally merge an override file and CLI args.

    Args:
        override_path: Path to a YAML file whose values take priority over the
            defaults.  May be ``None`` to use defaults only.
        cli_overrides: Flat ``dict`` of dot-separated keys to values, e.g.
            ``{"training.max_steps": 1000}``.  Applied last (highest priority).

    Returns:
        Fully-merged configuration dictionary.
    """
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)
    logger.info("Loaded base config from %s", DEFAULT_CONFIG_PATH)

    if override_path is not None:
        override_path = Path(override_path)
        with open(override_path, "r", encoding="utf-8") as fh:
            overrides: dict[str, Any] = yaml.safe_load(fh) or {}
        config = _deep_merge(config, overrides)
        logger.info("Merged override config from %s", override_path)

    if cli_overrides:
        for dotted_key, value in cli_overrides.items():
            keys = dotted_key.split(".")
            target = config
            for k in keys[:-1]:
                target = target.setdefault(k, {})
            target[keys[-1]] = value
        logger.info("Applied %d CLI overrides", len(cli_overrides))

    return config
