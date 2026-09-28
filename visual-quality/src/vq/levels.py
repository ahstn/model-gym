"""Q-Align level vocabulary, eval prompts, and answer stems (shared by scoring and training)."""

from __future__ import annotations

LEVELS: tuple[str, ...] = ("excellent", "good", "fair", "poor", "bad")
WEIGHTS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0)

TASKS: tuple[str, ...] = ("iqa", "iaa", "vqa")

# Fixed eval prompt per task: the Q-Align / Q-ReAlign model-card prompts.
EVAL_PROMPTS: dict[str, str] = {
    "iqa": "How would you rate the quality of this image?",
    "iaa": "How would you rate the aesthetics of this image?",
    "vqa": "How would you rate the quality of this video?",
}

STEMS: dict[str, str] = {
    "iqa": "The quality of the image is",
    "iaa": "The aesthetics of the image is",
    "vqa": "The quality of the video is",
}
