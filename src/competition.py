"""Competition compliance guards for WAXAL ASR.

Enforces that the repository only loads data from the official Zindi
WAXAL competition dataset (google/WaxalNLP) and only for the three
competition languages (Luganda, Lingala, Shona).

These guards exist to prevent accidental misuse — loading unsanctioned
datasets, non-competition languages, or invalid splits.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Competition constants
# ---------------------------------------------------------------------------

COMPETITION_MODE: bool = True
"""Global flag.  When ``True``, all dataset and language guards are active."""

ALLOWED_DATASET_ID: str = "google/WaxalNLP"
"""The only HuggingFace dataset identifier permitted in competition mode."""

ALLOWED_LANGUAGES: set[str] = {"lug", "lin", "sna"}
"""ISO 639-3 codes for the three competition languages."""

ALLOWED_SPLITS: set[str] = {"train", "validation", "test"}
"""Valid dataset split names."""

LANGUAGE_NAMES: dict[str, str] = {
    "lug": "Luganda",
    "lin": "Lingala",
    "sna": "Shona",
}
"""Human-readable names for logging."""


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_dataset_id(dataset_id: str) -> None:
    """Assert that *dataset_id* is the official competition dataset.

    Args:
        dataset_id: The HuggingFace dataset identifier to validate.

    Raises:
        ValueError: If *dataset_id* is not ``google/WaxalNLP``.
    """
    if not COMPETITION_MODE:
        return
    if dataset_id != ALLOWED_DATASET_ID:
        raise ValueError(
            f"Competition mode is active.  "
            f"Only dataset_id='{ALLOWED_DATASET_ID}' is permitted, "
            f"but got '{dataset_id}'.  "
            f"Set COMPETITION_MODE = False in src/competition.py to disable this check."
        )


def validate_language(language: str) -> None:
    """Assert that *language* is one of the three competition languages.

    Args:
        language: ISO 639-3 language code.

    Raises:
        ValueError: If *language* is not in ``ALLOWED_LANGUAGES``.
    """
    if not COMPETITION_MODE:
        return
    if language not in ALLOWED_LANGUAGES:
        raise ValueError(
            f"Competition mode is active.  "
            f"Only languages {sorted(ALLOWED_LANGUAGES)} are permitted, "
            f"but got '{language}'.  "
            f"The WAXAL challenge covers Luganda (lug), Lingala (lin), and Shona (sna) only."
        )


def validate_split(split: str) -> None:
    """Assert that *split* is a valid dataset split name.

    Args:
        split: The split name (``"train"``, ``"validation"``, or ``"test"``).

    Raises:
        ValueError: If *split* is not in ``ALLOWED_SPLITS``.
    """
    if not COMPETITION_MODE:
        return
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Competition mode is active.  "
            f"Only splits {sorted(ALLOWED_SPLITS)} are permitted, "
            f"but got '{split}'."
        )


def validate_languages_list(languages: list[str]) -> None:
    """Validate every language in a list.

    Args:
        languages: List of ISO 639-3 codes.

    Raises:
        ValueError: If any language is not permitted.
    """
    for lang in languages:
        validate_language(lang)


def validate_config(config: dict[str, Any]) -> None:
    """Run all competition validations against a merged config dict.

    Call this once after loading config and before any data operations.

    Args:
        config: Full merged configuration dictionary.

    Raises:
        ValueError: On any compliance violation.
    """
    if not COMPETITION_MODE:
        logger.warning("COMPETITION_MODE is disabled — skipping config validation.")
        return

    dataset_cfg = config.get("dataset", {})
    dataset_id = dataset_cfg.get("dataset_id", "")
    languages = dataset_cfg.get("languages", [])

    validate_dataset_id(dataset_id)
    validate_languages_list(languages)

    logger.info("Competition compliance check PASSED")
    logger.info("  Dataset : %s", dataset_id)
    logger.info("  Languages: %s", ", ".join(f"{l} ({LANGUAGE_NAMES.get(l, l)})" for l in languages))


# ---------------------------------------------------------------------------
# Dataset load summary (printed before training)
# ---------------------------------------------------------------------------

def log_dataset_load_summary(
    dataset_id: str,
    language: str,
    split: str,
    streaming: bool = True,
) -> None:
    """Log a structured summary of a dataset load operation.

    Args:
        dataset_id: HuggingFace dataset identifier.
        language: Language code.
        split: Split name.
        streaming: Whether streaming mode is used.
    """
    logger.info(
        "  Dataset  : %s\n"
        "  Language : %s (%s)\n"
        "  Split    : %s\n"
        "  Streaming: %s",
        dataset_id,
        language,
        LANGUAGE_NAMES.get(language, language),
        split,
        streaming,
    )


# ---------------------------------------------------------------------------
# Dataset audit report
# ---------------------------------------------------------------------------

def generate_dataset_audit(
    config: dict[str, Any],
    output_path: str | Path = "reports/dataset_audit.md",
) -> Path:
    """Generate a markdown audit report documenting all data sources.

    Args:
        config: Full merged configuration dictionary.
        output_path: Path for the report file.

    Returns:
        Path to the generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_cfg = config.get("dataset", {})
    dataset_id = dataset_cfg.get("dataset_id", "unknown")
    languages = dataset_cfg.get("languages", [])
    sample_rate = dataset_cfg.get("sample_rate", 16000)
    streaming = dataset_cfg.get("streaming", True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# WAXAL ASR — Dataset Audit Report",
        "",
        f"**Generated:** {now}",
        f"**Competition mode:** {'ENABLED' if COMPETITION_MODE else 'DISABLED'}",
        "",
        "---",
        "",
        "## Dataset Source",
        "",
        f"- **Dataset ID:** `{dataset_id}`",
        f"- **Allowed dataset:** `{ALLOWED_DATASET_ID}`",
        f"- **Match:** {'YES' if dataset_id == ALLOWED_DATASET_ID else 'NO'}",
        "",
        "## Languages Loaded",
        "",
        "| Code | Language | Allowed |",
        "|------|----------|---------|",
    ]

    for lang in languages:
        name = LANGUAGE_NAMES.get(lang, "Unknown")
        allowed = "YES" if lang in ALLOWED_LANGUAGES else "NO"
        lines.append(f"| `{lang}` | {name} | {allowed} |")

    lines.extend([
        "",
        "## Splits",
        "",
        "| Split | Loaded |",
        "|-------|--------|",
        "| train | YES |",
        "| validation | YES |",
        "| test | YES |",
        "",
        "## Audio Configuration",
        "",
        f"- **Sample rate:** {sample_rate} Hz",
        f"- **Streaming:** {streaming}",
        "",
        "## Compliance Status",
        "",
    ])

    all_ok = (
        dataset_id == ALLOWED_DATASET_ID
        and all(l in ALLOWED_LANGUAGES for l in languages)
    )

    if all_ok:
        lines.append("**PASSED** — All data sources comply with competition rules.")
    else:
        lines.append("**FAILED** — Non-compliant data sources detected.")
        if dataset_id != ALLOWED_DATASET_ID:
            lines.append(f"- Dataset ID `{dataset_id}` is not the allowed dataset.")
        for lang in languages:
            if lang not in ALLOWED_LANGUAGES:
                lines.append(f"- Language `{lang}` is not a competition language.")

    lines.extend([
        "",
        "---",
        "",
        "*Generated automatically by `src/competition.py`*",
    ])

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("Dataset audit report saved to %s", output_path)
    return output_path
