"""Subtitle export (SRT/VTT) and enhanced history for MacWhisper.

Generates subtitle files from word-level timestamps and saves
enriched recording history.
"""

from __future__ import annotations

import json
import os
import datetime

from asr_backend import WordTimestamp


# ── SRT export ───────────────────────────────────────────

def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _group_words_into_lines(
    words: list[WordTimestamp],
    max_words_per_line: int = 12,
    max_line_width: int = 42,
    max_gap_s: float = 1.5,
) -> list[list[WordTimestamp]]:
    """Group words into subtitle lines based on count, width, and pauses."""
    if not words:
        return []

    lines: list[list[WordTimestamp]] = []
    current_line: list[WordTimestamp] = []
    current_width = 0

    for w in words:
        word_text = w.word.strip()
        if not word_text:
            continue

        # Start new line on large gap
        if current_line and (w.start - current_line[-1].end) > max_gap_s:
            lines.append(current_line)
            current_line = []
            current_width = 0

        # Start new line on word/width limit
        new_width = current_width + len(word_text) + (1 if current_line else 0)
        if current_line and (
            len(current_line) >= max_words_per_line
            or new_width > max_line_width
        ):
            lines.append(current_line)
            current_line = []
            current_width = 0

        current_line.append(w)
        current_width += len(word_text) + (1 if len(current_line) > 1 else 0)

    if current_line:
        lines.append(current_line)

    return lines


def export_srt(
    words: list[WordTimestamp],
    path: str,
    max_words_per_line: int = 12,
    max_line_width: int = 42,
) -> str:
    """Export word timestamps as an SRT subtitle file.

    Returns the path written.
    """
    lines = _group_words_into_lines(words, max_words_per_line, max_line_width)

    parts = []
    for i, line_words in enumerate(lines, 1):
        start = _format_srt_time(line_words[0].start)
        end = _format_srt_time(line_words[-1].end)
        text = "".join(w.word for w in line_words).strip()
        parts.append(f"{i}\n{start} --> {end}\n{text}\n")

    content = "\n".join(parts)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def export_vtt(
    words: list[WordTimestamp],
    path: str,
    max_words_per_line: int = 12,
    max_line_width: int = 42,
) -> str:
    """Export word timestamps as a WebVTT subtitle file.

    Returns the path written.
    """
    lines = _group_words_into_lines(words, max_words_per_line, max_line_width)

    parts = ["WEBVTT\n"]
    for line_words in lines:
        start = _format_vtt_time(line_words[0].start)
        end = _format_vtt_time(line_words[-1].end)
        text = "".join(w.word for w in line_words).strip()
        parts.append(f"{start} --> {end}\n{text}\n")

    content = "\n".join(parts)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ── Enhanced history ─────────────────────────────────────

def save_enhanced_history(
    history_dir: str,
    audio_file: str | None,
    words: list[WordTimestamp],
    text: str,
    model: str,
    duration_s: float,
    language: str = "",
):
    """Save enriched recording history: transcript JSON + SRT.

    Files saved alongside the audio in history_dir:
      - transcripts.jsonl  (appended)
      - <timestamp>.srt    (if words available)
      - <timestamp>.json   (word-level timestamps)
    """
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Append to transcript log
    transcript_log = os.path.join(history_dir, "transcripts.jsonl")
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "audio_file": audio_file,
        "model": model,
        "duration_s": round(duration_s, 1),
        "language": language,
        "text": text,
        "word_count": len(words),
    }
    with open(transcript_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # Save word-level timestamps as JSON
    if words:
        words_path = os.path.join(history_dir, f"{timestamp}_words.json")
        words_data = [
            {"word": w.word, "start": w.start, "end": w.end,
             "probability": w.probability}
            for w in words
        ]
        with open(words_path, "w", encoding="utf-8") as f:
            json.dump(words_data, f, ensure_ascii=False, indent=2)

        # Generate SRT
        srt_path = os.path.join(history_dir, f"{timestamp}.srt")
        export_srt(words, srt_path)
