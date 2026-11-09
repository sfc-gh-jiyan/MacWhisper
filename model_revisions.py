"""Pinned Hugging Face commits for the models MacWhisper loads.

Both sidecars import this, so a repo name always resolves to the same files
(no silent upstream change). A repo not listed here (user override via
MACWHISPER_QWEN_MODEL or the config) loads the latest revision, with a warning.

To update a pin: run the model once unpinned, then copy the commit from
~/.cache/huggingface/hub/models--<org>--<name>/refs/main into this dict.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MODEL_REVISIONS = {
    "mlx-community/Qwen3-ASR-0.6B-bf16": "eae2b51f96265328f1e7beced788adb0e4536f92",
    "mlx-community/Qwen3-ASR-1.7B-bf16": "e1f6c266914abc5a46e8756e02580f834a6cf8a7",
    "mlx-community/Qwen3-4B-Instruct-2507-4bit": "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
}


def revision_for(repo: str) -> str | None:
    """Pinned commit for `repo`, or None (unpinned, logged) if it is not known."""
    rev = MODEL_REVISIONS.get(repo)
    if rev is None:
        logger.warning("Model %s has no pinned revision; loading the latest upstream files", repo)
    return rev
