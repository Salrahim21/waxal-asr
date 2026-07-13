"""Audio and text preprocessing for WAXAL ASR.

Handles chat-message formatting that the Gemma 3n model expects.  Each audio
example is wrapped into a ``messages`` list with system, user (audio + text),
and assistant (transcription) turns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Single-example formatter
# ---------------------------------------------------------------------------

def format_for_chat(
    example: Mapping[str, Any],
    system_message: str,
    user_message: str,
) -> dict[str, Any]:
    """Convert a decoded HuggingFace example into a chat-formatted message list.

    The ``audio`` field is expected to already be decoded by the ``datasets``
    library (i.e., it contains ``array`` and ``sampling_rate`` keys).

    Args:
        example: Dict with ``audio`` (decoded) and ``transcription`` keys.
        system_message: System prompt text.
        user_message: User prompt text.

    Returns:
        A copy of *example* with an additional ``messages`` key.
    """
    audio_array = example["audio"]["array"]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": user_message},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": str(example["transcription"])}
            ],
        },
    ]
    return {**example, "messages": messages}


# ---------------------------------------------------------------------------
# Batch operator (for datasets.map)
# ---------------------------------------------------------------------------

def format_batch(
    batch: dict[str, list[Any]],
    system_message: str,
    user_message: str,
) -> dict[str, list[Any]]:
    """Apply :func:`format_for_chat` to a batched ``datasets.map`` call.

    Args:
        batch: Batched dict where each value is a list of per-example values.
        system_message: System prompt text.
        user_message: User prompt text.

    Returns:
        The batch dict augmented with a ``messages`` list.
    """
    num = len(batch["transcription"])
    examples = [{k: batch[k][i] for k in batch} for i in range(num)]
    formatted = [
        format_for_chat(ex, system_message=system_message, user_message=user_message)
        for ex in examples
    ]
    return {k: [ex[k] for ex in formatted] for k in formatted[0]}
